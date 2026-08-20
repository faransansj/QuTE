# CC-NQE P4.6-B Track-A Report

Validation-only winner: **A3**
Track-A verdict: **MIXED-DATA-EFFECT**

Pre-specified heuristic (not a statistical significance claim): balanced-score range < 0.01 => NO-CLEAR-DATA-EFFECT; otherwise A1/A2 winner => CIRCUIT-COVERAGE-DOMINANT, A3 => MIXED-DATA-EFFECT, A4/A5 => PROBE-COVERAGE-DOMINANT.

All primary comparisons use each arm's single best-balanced checkpoint. Per-metric best values are secondary diagnostics. No sealed split was loaded.

## A1

- exact circuits / structural signatures: 1,000,000 / 376,414
- unique k2/k3/k4 motifs: {'k2': 930, 'k3': 27814, 'k4': 338486}
- parameter bins: [0, 1, 2, 3, 6, 7, 8, 9]; interactions: {'2->1': 35163, '0->3': 34789, '3->1': 35321, '0->2': 32708, '1->3': 37918, '1->0': 36048, '2->3': 40136, '1->2': 35075, '2->0': 35821, '3->0': 34127, '3->2': 39056}
- probes/circuit: 1; probe statistics: {"probe_reuse_multiplicity": {"maximum": 2095, "mean": 1.999132376548578, "minimum": 1, "reused_state_count": 295}, "projective_probe_diversity": {"definition": "adjacent deterministic sample squared overlaps |<psi_i|psi_j>|^2; lower means more diverse", "maximum": 0.8916500210762024, "mean": 0.06395763903856277, "median": 0.021038327366113663, "p95": 0.28011181950569153, "sample_pairs": 4095}, "state_family_counts": {"product": 500000, "random-local": 500000}, "unique_probe_states": 500217}
- primary checkpoint: {"metrics": {"balanced_validation": 0.3090827572482845, "composition_ood_validation": 0.16861730532643074, "depth_ood_validation": 0.2633027588017285, "iid_validation": 0.49532820761669427, "parameter_ood_validation": 0.6556076469520727, "state_ood_validation": 0.40008926515777904, "step": 10000}, "path": "artifacts/cc_nqe_p4_6/track_a/checkpoints/A1-best-balanced.pt", "selection": "best balanced validation"}
- final train / gap: 0.53618819 / 0.04085998
- best validation (secondary): {"composition_ood_validation": {"step": 5500, "value": 0.17903411456306154}, "depth_ood_validation": {"step": 9500, "value": 0.26340615935623646}, "iid_validation": {"step": 2000, "value": 0.5527147563795248}, "parameter_ood_validation": {"step": 9500, "value": 0.6557371119658152}, "state_ood_validation": {"step": 10000, "value": 0.40008926515777904}}
- final validation: {"composition_ood_validation": 0.16861730532643074, "depth_ood_validation": 0.2633027588017285, "iid_validation": 0.49532820761669427, "parameter_ood_validation": 0.6556076469520727, "state_ood_validation": 0.40008926515777904}
- exposure: updates=10000, pairs=10240000, unique-pair=999958 (0.999958), pair-epochs=10.2400, circuit=10240000, mean/circuit=10.2400, probe=10240000, mean/probe=20.4711
- samples/wall/rate: 10240000 / 310.75s / 32952.48 samples/s

## A2

- exact circuits / structural signatures: 250,000 / 113,609
- unique k2/k3/k4 motifs: {'k2': 930, 'k3': 26486, 'k4': 145287}
- parameter bins: [0, 1, 2, 3, 6, 7, 8, 9]; interactions: {'2->1': 8855, '0->3': 8837, '3->1': 8904, '0->2': 8235, '1->3': 9441, '1->0': 8993, '2->3': 10013, '1->2': 8735, '2->0': 8933, '3->0': 8500, '3->2': 9789}
- probes/circuit: 4; probe statistics: {"probe_reuse_multiplicity": {"maximum": 2074, "mean": 1.9990804230054176, "minimum": 1, "reused_state_count": 282}, "projective_probe_diversity": {"definition": "adjacent deterministic sample squared overlaps |<psi_i|psi_j>|^2; lower means more diverse", "maximum": 1.0, "mean": 0.06805090606212616, "median": 0.020961984992027283, "p95": 0.27633118629455566, "sample_pairs": 4095}, "state_family_counts": {"product": 500000, "random-local": 500000}, "unique_probe_states": 500230}
- primary checkpoint: {"metrics": {"balanced_validation": 0.313231829372752, "composition_ood_validation": 0.16097929258830845, "depth_ood_validation": 0.25531865702942014, "iid_validation": 0.5233975385005275, "parameter_ood_validation": 0.6008848287165165, "state_ood_validation": 0.40620827364424866, "step": 7500}, "path": "artifacts/cc_nqe_p4_6/track_a/checkpoints/A2-best-balanced.pt", "selection": "best balanced validation"}
- final train / gap: 0.55303419 / 0.02355461
- best validation (secondary): {"composition_ood_validation": {"step": 8000, "value": 0.16591240760559836}, "depth_ood_validation": {"step": 4000, "value": 0.2587168758036569}, "iid_validation": {"step": 9500, "value": 0.5303947484741608}, "parameter_ood_validation": {"step": 5500, "value": 0.6266983623305956}, "state_ood_validation": {"step": 10000, "value": 0.4091774507736166}}
- final validation: {"composition_ood_validation": 0.16066307310635844, "depth_ood_validation": 0.24854585016146302, "iid_validation": 0.5294795759643117, "parameter_ood_validation": 0.5937456836303076, "state_ood_validation": 0.4091774507736166}
- exposure: updates=10000, pairs=10240000, unique-pair=999958 (0.999958), pair-epochs=10.2400, circuit=10240000, mean/circuit=40.9600, probe=10240000, mean/probe=20.4706
- samples/wall/rate: 10240000 / 304.74s / 33602.54 samples/s

## A3

- exact circuits / structural signatures: 62,500 / 33,413
- unique k2/k3/k4 motifs: {'k2': 930, 'k3': 20996, 'k4': 51148}
- parameter bins: [0, 1, 2, 3, 6, 7, 8, 9]; interactions: {'2->1': 2217, '0->3': 2207, '3->1': 2305, '0->2': 2051, '1->3': 2353, '1->0': 2230, '2->3': 2521, '1->2': 2149, '2->0': 2248, '3->0': 2171, '3->2': 2385}
- probes/circuit: 16; probe statistics: {"probe_reuse_multiplicity": {"maximum": 2067, "mean": 1.9990484529364023, "minimum": 1, "reused_state_count": 274}, "projective_probe_diversity": {"definition": "adjacent deterministic sample squared overlaps |<psi_i|psi_j>|^2; lower means more diverse", "maximum": 1.0, "mean": 0.06314011663198471, "median": 0.01978476718068123, "p95": 0.2650907039642334, "sample_pairs": 4095}, "state_family_counts": {"product": 500000, "random-local": 500000}, "unique_probe_states": 500238}
- primary checkpoint: {"metrics": {"balanced_validation": 0.3232958903665551, "composition_ood_validation": 0.16905439633410424, "depth_ood_validation": 0.25643077911809087, "iid_validation": 0.5444024956474701, "parameter_ood_validation": 0.5866255496318141, "state_ood_validation": 0.4055997282266617, "step": 9500}, "path": "artifacts/cc_nqe_p4_6/track_a/checkpoints/A3-best-balanced.pt", "selection": "best balanced validation"}
- final train / gap: 0.55881673 / 0.01304128
- best validation (secondary): {"composition_ood_validation": {"step": 9500, "value": 0.16905439633410424}, "depth_ood_validation": {"step": 7500, "value": 0.2605464421212673}, "iid_validation": {"step": 10000, "value": 0.5457754557331403}, "parameter_ood_validation": {"step": 5000, "value": 0.6035696578522524}, "state_ood_validation": {"step": 8500, "value": 0.40704054571688175}}
- final validation: {"composition_ood_validation": 0.16807555983541533, "depth_ood_validation": 0.25529620610177517, "iid_validation": 0.5457754557331403, "parameter_ood_validation": 0.5872243760774533, "state_ood_validation": 0.4042257151256005}
- exposure: updates=10000, pairs=10240000, unique-pair=999958 (0.999958), pair-epochs=10.2400, circuit=10240000, mean/circuit=163.8400, probe=10240000, mean/probe=20.4703
- samples/wall/rate: 10240000 / 305.64s / 33503.09 samples/s

## A4

- exact circuits / structural signatures: 58,824 / 31,665
- unique k2/k3/k4 motifs: {'k2': 930, 'k3': 20629, 'k4': 48627}
- parameter bins: [0, 1, 2, 3, 6, 7, 8, 9]; interactions: {'2->1': 2082, '0->3': 2076, '3->1': 2161, '0->2': 1924, '1->3': 2242, '1->0': 2091, '2->3': 2374, '1->2': 2024, '2->0': 2114, '3->0': 2043, '3->2': 2223}
- probes/circuit: 17; probe statistics: {"probe_reuse_multiplicity": {"maximum": 2086, "mean": 1.9990364721286569, "minimum": 1, "reused_state_count": 271}, "projective_probe_diversity": {"definition": "adjacent deterministic sample squared overlaps |<psi_i|psi_j>|^2; lower means more diverse", "maximum": 0.8953341841697693, "mean": 0.06268220394849777, "median": 0.019736409187316895, "p95": 0.26938578486442566, "sample_pairs": 4095}, "state_family_counts": {"product": 500004, "random-local": 500004}, "unique_probe_states": 500245}
- primary checkpoint: {"metrics": {"balanced_validation": 0.3165291210041485, "composition_ood_validation": 0.1606642738527929, "depth_ood_validation": 0.24748443230055273, "iid_validation": 0.5414386568590999, "parameter_ood_validation": 0.6475287713110447, "state_ood_validation": 0.3768165037035942, "step": 8500}, "path": "artifacts/cc_nqe_p4_6/track_a/checkpoints/A4-best-balanced.pt", "selection": "best balanced validation"}
- final train / gap: 0.55213249 / 0.01501940
- best validation (secondary): {"composition_ood_validation": {"step": 7000, "value": 0.16416538555252677}, "depth_ood_validation": {"step": 7500, "value": 0.25782431359402835}, "iid_validation": {"step": 8500, "value": 0.5414386568590999}, "parameter_ood_validation": {"step": 7500, "value": 0.656750850379467}, "state_ood_validation": {"step": 5500, "value": 0.3921276753147443}}
- final validation: {"composition_ood_validation": 0.16180460946634412, "depth_ood_validation": 0.246862432686612, "iid_validation": 0.537113086010019, "parameter_ood_validation": 0.648627508431673, "state_ood_validation": 0.38238280825316906}
- exposure: updates=10000, pairs=10240000, unique-pair=999983 (0.999975), pair-epochs=10.2399, circuit=10240000, mean/circuit=174.0786, probe=10240000, mean/probe=20.4700
- samples/wall/rate: 10240000 / 305.81s / 33484.45 samples/s

## A5

- exact circuits / structural signatures: 15,625 / 9,651
- unique k2/k3/k4 motifs: {'k2': 929, 'k3': 11981, 'k4': 14767}
- parameter bins: [0, 1, 2, 3, 6, 7, 8, 9]; interactions: {'2->1': 572, '0->3': 530, '3->1': 527, '0->2': 509, '1->3': 599, '1->0': 542, '2->3': 648, '1->2': 544, '2->0': 563, '3->0': 565, '3->2': 580}
- probes/circuit: 64; probe statistics: {"probe_reuse_multiplicity": {"maximum": 2065, "mean": 1.9990804230054176, "minimum": 1, "reused_state_count": 282}, "projective_probe_diversity": {"definition": "adjacent deterministic sample squared overlaps |<psi_i|psi_j>|^2; lower means more diverse", "maximum": 0.8953341841697693, "mean": 0.06237903982400894, "median": 0.02090851403772831, "p95": 0.26306822896003723, "sample_pairs": 4095}, "state_family_counts": {"product": 500000, "random-local": 500000}, "unique_probe_states": 500230}
- primary checkpoint: {"metrics": {"balanced_validation": 0.3072606130477248, "composition_ood_validation": 0.17465935445701083, "depth_ood_validation": 0.2508786079706624, "iid_validation": 0.49624387671550113, "parameter_ood_validation": 0.5328183804328243, "state_ood_validation": 0.34594734633962315, "step": 8500}, "path": "artifacts/cc_nqe_p4_6/track_a/checkpoints/A5-best-balanced.pt", "selection": "best balanced validation"}
- final train / gap: 0.59621739 / 0.09954083
- best validation (secondary): {"composition_ood_validation": {"step": 5000, "value": 0.1861899538586537}, "depth_ood_validation": {"step": 8500, "value": 0.2508786079706624}, "iid_validation": {"step": 10000, "value": 0.4966765645270546}, "parameter_ood_validation": {"step": 6000, "value": 0.5378204379230738}, "state_ood_validation": {"step": 10000, "value": 0.3461379312599699}}
- final validation: {"composition_ood_validation": 0.17486986851630112, "depth_ood_validation": 0.24275718233548105, "iid_validation": 0.4966765645270546, "parameter_ood_validation": 0.5341305689265331, "state_ood_validation": 0.3461379312599699}
- exposure: updates=10000, pairs=10240000, unique-pair=999958 (0.999958), pair-epochs=10.2400, circuit=10240000, mean/circuit=655.3600, probe=10240000, mean/probe=20.4706
- samples/wall/rate: 10240000 / 306.12s / 33450.62 samples/s

## Audits

- circuit nesting: {'passed': True, 'relation': 'A5 subset A4 subset A3 subset A2 subset A1', 'method': 'ordered master-pool prefixes'}
- probe nesting: {'passed': True, 'relation': 'P1(C) subset P4(C) subset P16(C) subset P17(C) subset P64(C)', 'method': 'ordered deterministic probe prefixes; 1000-circuit regeneration check'}
- distribution control: {"A1": {"matched_within_0.02": true, "maximum_fraction_differences_from_A1": {"depth": 0.0, "parameter_bin": 0.0, "primitive_gate": 0.0, "qubit_interaction": 0.0, "state_family": 0.0}}, "A2": {"matched_within_0.02": true, "maximum_fraction_differences_from_A1": {"depth": 2.000000000002e-06, "parameter_bin": 0.000600815347850947, "primitive_gate": 0.0005419237092754992, "qubit_interaction": 0.0012361562535603693, "state_family": 0.0}}, "A3": {"matched_within_0.02": true, "maximum_fraction_differences_from_A1": {"depth": 1.0000000000010001e-05, "parameter_bin": 0.001525193718666093, "primitive_gate": 0.0013656606905220192, "qubit_interaction": 0.0036471184523169803, "state_family": 0.0}}, "A4": {"matched_within_0.02": true, "maximum_fraction_differences_from_A1": {"depth": 6.666666666488297e-07, "parameter_bin": 0.0016282444633806958, "primitive_gate": 0.0009227740946187446, "qubit_interaction": 0.0033988120382631576, "state_family": 0.0}}, "A5": {"matched_within_0.02": true, "maximum_fraction_differences_from_A1": {"depth": 5.2999999999997494e-05, "parameter_bin": 0.003043504039435166, "primitive_gate": 0.001117027078924776, "qubit_interaction": 0.005294691952681355, "state_family": 0.0}}}
- sealed-test access count: 0

## STOP

Track A complete. Track B/C and seeds 2027/2028 were not run.
