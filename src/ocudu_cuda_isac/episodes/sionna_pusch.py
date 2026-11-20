"""Second transmit reference for a grant: the pinned Sionna NR PUSCH generator.

Offline only. It needs the full environment (PyTorch and Sionna) and runs on the CPU. The first
reference is the pinned OCUDU transmit chain, reached through the native fixture adapter.

Sionna places its DM-RS sequence from the first subcarrier of the configured bandwidth part, and its
PUSCH always fills that bandwidth part. A true PUSCH references the sequence to CRB 0. Two
configurations therefore produce one aligned reference:

  data   a bandwidth part that is exactly the allocation. Its data symbols depend only on the
         transport block, the RNTI, n_ID, the MCS and the number of resource elements.
  dmrs   a bandwidth part from CRB 0 to the end of the allocation. Its DM-RS on the allocated
         blocks is the CRB 0 referenced sequence.

Sionna scales DM-RS by sqrt(2) for two CDM groups without data. The grant, like the pinned
receiver, uses 10**(3/20). The reference is rescaled to the grant's amplitude and the factor is
reported.
"""

from .grant import GrantError, SYMBOLS_PER_SLOT

MCS_TABLE = {"qam64": 1, "qam256": 2, "qam64LowSe": 3}


def parameters(grant):
    """The two Sionna configurations for a grant, as plain values. Needs no Sionna."""
    common = {
        "carrier": {"n_cell_id": grant.n_cell_id, "subcarrier_spacing": grant.scs_khz,
                    "n_size_grid": grant.grid_size_rb, "slot_number": grant.slot, "frame_number": grant.sfn,
                    "cyclic_prefix": "normal"},
        "dmrs": {"config_type": grant.dmrs["config_type"], "length": grant.dmrs["length"],
                 "type_a_position": grant.dmrs["type_a_position"],
                 "additional_position": grant.dmrs["additional_position"],
                 "dmrs_port_set": list(grant.dmrs["ports"]),
                 "n_id": [grant.dmrs["scrambling_id"], grant.dmrs["scrambling_id"]], "n_scid": grant.dmrs["n_scid"],
                 "num_cdm_groups_without_data": grant.dmrs["cdm_groups_without_data"]},
        "tb": {"channel_type": "PUSCH", "mcs_table": MCS_TABLE[grant.mcs_table], "mcs_index": grant.mcs_index,
               "n_id": grant.n_id},
        "pusch": {"n_start_bwp": 0, "num_layers": grant.layers, "num_antenna_ports": grant.antenna_ports,
                  "mapping_type": grant.mapping_type, "symbol_allocation": [grant.symbol_start, grant.symbol_count],
                  "n_rnti": grant.rnti, "precoding": "non-codebook"},
    }
    return {"common": common, "data_n_size_bwp": grant.prb_count,
            "dmrs_n_size_bwp": grant.crb_start + grant.prb_count,
            "dmrs_first_subcarrier": grant.subcarrier_start}


def reference(grant, device="cpu"):
    """Generate the aligned Sionna transmit grid for a grant.

    Returns a dict: x (complex values for antenna_ports * 14 * grid_subcarriers, last axis fastest,
    carrier coordinates), tb_size, coded_bits, dmrs_symbols, dmrs_scale_applied,
    own_dmrs_is_crb0_referenced and versions.
    """
    try:
        import numpy as np
        import torch
        import sionna
        from sionna.phy.nr import CarrierConfig, PUSCHConfig, PUSCHDMRSConfig, PUSCHTransmitter, TBConfig
    except ImportError as error:
        raise GrantError(f"the Sionna reference needs the full environment: {error}") from None
    values = parameters(grant)
    common = values["common"]

    def config(n_size_bwp):
        dmrs = dict(common["dmrs"], n_id=tuple(common["dmrs"]["n_id"]))
        return PUSCHConfig(carrier_config=CarrierConfig(**common["carrier"]),
                           pusch_dmrs_config=PUSCHDMRSConfig(**dmrs), tb_config=TBConfig(**common["tb"]),
                           n_size_bwp=n_size_bwp, **common["pusch"])

    data_config = config(values["data_n_size_bwp"])
    if data_config.tb_size != grant.tbs_bits:
        raise GrantError(f"Sionna computes a transport block of {data_config.tb_size} bits, the grant {grant.tbs_bits}")
    if list(data_config.dmrs_symbol_indices) != list(grant.dmrs["symbols"]):
        raise GrantError(f"Sionna places DM-RS on symbols {list(data_config.dmrs_symbol_indices)}, "
                         f"the grant on {grant.dmrs['symbols']}")
    transmitter = PUSCHTransmitter(data_config, return_bits=False, output_domain="freq", precision="double",
                                   device=device)
    bits = torch.tensor(grant.tb_bits(), dtype=torch.float64, device=device).reshape(1, 1, -1)
    with torch.no_grad():
        generated = transmitter(bits).cpu().numpy()[0, 0, 0]  # [symbol, subcarrier of the allocation]
    # [port, subcarrier from CRB 0, symbol] -> [symbol, subcarrier of the allocation]
    dmrs = config(values["dmrs_n_size_bwp"]).dmrs_grid[0][values["dmrs_first_subcarrier"]:, :].T
    scale = grant.derived()["tx_scale"]["dmrs_amplitude"] / float(data_config.dmrs.beta)

    start, count = grant.subcarrier_start, grant.subcarrier_count
    grid = np.zeros((SYMBOLS_PER_SLOT, grant.grid_subcarriers), dtype=np.complex128)
    for symbol in range(grant.symbol_start, grant.symbol_start + grant.symbol_count):
        row = generated[symbol].copy()
        if symbol in grant.dmrs["symbols"]:
            # Pilots come from the CRB 0 referenced configuration. Data that shares the symbol stays.
            pilots = dmrs[symbol] != 0
            row[pilots] = scale * dmrs[symbol][pilots]
        grid[symbol, start:start + count] = row
    own = generated[grant.dmrs["symbols"], :]
    return {
        "x": [complex(value) for value in grid.reshape(-1)],
        "tb_size": int(data_config.tb_size), "coded_bits": int(data_config.num_coded_bits),
        "dmrs_symbols": [int(symbol) for symbol in data_config.dmrs_symbol_indices],
        "dmrs_scale_applied": scale, "sionna_dmrs_beta": float(data_config.dmrs.beta),
        "own_dmrs_is_crb0_referenced": bool(np.allclose(own, dmrs[grant.dmrs["symbols"], :])),
        "configurations": values,
        "versions": {"sionna": sionna.__version__, "torch": torch.__version__, "numpy": np.__version__},
        "device": device, "precision": "double",
    }
