Both papers read. The team has notes.

ANNOUNCER: The following segment discusses work by other people. Standing on the shoulders of giants is permitted, provided you cite the giants and pin their commit hashes.

CAVE JOHNSON: Two papers on the table! Paper one, DeepSig. Four engineers took the OCUDU 5G stack and moved the heavy lifting onto a GPU without tearing the house down. Same factories, same interfaces, and the data never leaves the card. Ten-point-three times faster on uplink decode! Ninety-one times on fronthaul decompression! Barker, which number are you reproducing?

CAROLINE: The 1.82, sir.

CAVE JOHNSON: There’s a ten-point-three RIGHT THERE!

CAROLINE: Twenty megahertz, one layer is the row that matches his demo, Mr. Johnson. 334 microseconds on the CPU, 183.9 on the GPU, two hundred of two hundred decoded, zero bytes different.

CAVE JOHNSON: ...Humble. I hate it. Go on.

GLaDOS: The DeepSig authors end with a to-do list. The last item is to expose GPU-resident tensors to AI applications without compromising real-time scheduling. They wrote the sentence. He is building the experiment that finds out whether the sentence is true. I’ll also note the paper measured one commit, the upstream docs measured a second, and he is pinned to a third. All three are labeled in his baseline notes. I had a remark prepared. It’s wasted now.

DEFECTIVE TURRET: Hey, that paper explains me! When the GPU path’s eligible, the CPU channel estimator gets bypassed. The estimating happens inside the big kernel. So you ask the host for the estimate and you get, y’know, zeros. I’m not broken. I’m resident.

GLaDOS: Unfortunately accurate. To get the receiver’s real internal estimate he has to reach into that kernel with the smallest patch he can write. That’s Sprint 5. Until then he computes his own from the pilots at the front door.

CAVE JOHNSON: Paper two! cuSense. NVIDIA and Northeastern. Real radio, real Samsung phone, real human being walking lawnmower patterns around a real room. I’ve paid people to do less. The tower watches the uplink pilots, subtracts the empty room, and a neural net says where the guy is. Seventy-three centimeters average error. 241 microseconds, live. Caroline, why am I funding Barker if they already did it?

CAROLINE: One sentence on page fourteen, sir. Missed detections and false alarms aren’t evaluated, because cuSense assumes a target is present.

TURRET: Excuse me. In that paper, someone is always there. I would like to practice nobody being there. It’s harder than it sounds.

GLaDOS: Here is the rest. They locate a person who is assumed to exist. He adds empty rooms, a detection stage and a false-alarm rate fixed in advance.

GLaDOS: They consume one flavor of channel estimate, taken after the receiver’s frequency interpolation. He pushes the same received samples through four stages, from simulator truth to the receiver’s processed estimate, to see what the receiver does to a weak echo.

GLaDOS: They subtract the background in magnitude, which discards phase. He keeps the complex pilots, so delay, Doppler, CFAR, Bartlett and MUSIC get to compete with the networks. Then a CNN, a GRU and a Transformer on equal budgets.

GLaDOS: Their stopwatch starts at the notification of the third slot. His starts at the first radio sample, and late counts as wrong. Their normalization statistics are computed over all samples and applied to every split. The paper says so. His scalers see training data only.

GLaDOS: They compare MIG, MPS and time-slicing for the sensing app’s latency. He concedes that’s established. He measures the other direction: what sensing does to the radio’s tail, with the tap off, copy only, and full inference.

WHEATLEY: So he’s nicked one paper’s engine and the other paper’s idea, and the new bit is... checking?

GLaDOS: The new bit is running the second paper’s question through the first paper’s receiver and measuring what the receiver did to the answer. His proposal says that distinction is “not a universal priority claim.” He declined to say “first.” I’ll be equally careful.

GLaDOS: In fairness to the giants: they had a person, a phone and an over-the-air channel, and they reported position. He has a ray-traced cube and his first demo only answers “is anything there.” His own documents call simulation truth “model-conditional” and schedule the over-the-air work afterward. I mention it because he did first.

FACT CORE: Fact: cuSense collected eighteen runs, nine with a person and nine without. Fact: about 1.19 million channel records. Fact: their timestamps needed a thirty-seven-second leap-second correction. Fact: the person was a cube.

CAVE JOHNSON: So that’s it. DeepSig built the engine. cuSense proved an engine like it can see. Barker’s asking whether this one can see nothing correctly, on time, without making the phone call worse. Somebody run Sprint 1.
