/* The public header of the sensing core compiles as C99 (Sprint 3.6, selector sense_core). The
 * core is used from the PUSCH API of a receiver and from a dApp, and both sides speak C at their
 * boundary. This file is compiled with -std=c99 -Wpedantic -Werror and never linked. ASCII only.
 */

#include "sense_core/sense_core.h"

int sense_core_api_compiles_as_c(void)
{
  sense_options  options;
  sense_view     view;
  sense_answer   answer;
  sense_decision decision;
  sense_counters counters;
  options.api_version = SENSE_CORE_API_VERSION;
  view.memory         = SENSE_MEMORY_HOST_SHARED;
  answer.kind         = SENSE_ANSWER_READY;
  decision.status     = SENSE_DECISION_OK;
  counters.records    = 0;
  return (int)(options.api_version + (unsigned)view.memory + (unsigned)answer.kind + (unsigned)decision.status +
               (unsigned)counters.records + sizeof(sense_plan_info) + sizeof(sense_smoothed));
}
