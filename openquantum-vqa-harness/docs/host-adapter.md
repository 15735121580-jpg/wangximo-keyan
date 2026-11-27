# Writing a host adapter

Use the same original executable, including feedback and terminal scoring, for every arm. Use immutable task, hardware, policy and registered recipe objects. The general interfaces are Python calls. The MCP server exposes the bundled finite-grid job and native-host compiler tools.

| Entry point | Construction and result audit |
| --- | --- |
| `compile_direct(job, branch, adapter)` | Build one complete proposal from B and verify it. |
| `compile_bounded(job, branches, budget, adapter, incumbent=a1)` | Register an ordered tuple of unique recipes. Build at most the prefix of length `budget`, each from B. The audit records the recipe, stages and certified comparisons with B and the protected direct policy. |
| `direct_diagram(job, relations, verifier, max_rewrites)` | Follow the first unseen legal match in registered order, then assess the resulting complete program. |
| `search_diagram(job, relations, verifier, max_rewrites, max_candidates)` | Visit the breadth-first closure of distinct typed diagrams to the given depth and budget. The declared priority derivation supplies A1 unless its receipt is passed explicitly. A costlier intermediate can lead to a cheaper final program. The receipt reports construction, assessment, coverage and the winning derivation. |

## Implement the client

1. Construct `HostJob(task, hardware, host, stages, contract)`. Register protected losses in priority order. Represent higher-is-better quantities as losses. Model-based and device-observed quantities need separate evidence.
2. Implement `adapter.construct(job, branch)` to return the complete runnable policy. Implement `adapter.verifier(job)` to bind the trusted verifier to the frozen task and device.
3. Implement `verifier.assess(contract, policy)` to check semantics, native legality, fixed interfaces and each hard cap for the complete job. Return `Assessment("feasible", evidence)` for certified feasibility, `"infeasible"` for a demonstrated violation, and `"missing"` or `"invalid"` for unresolved evidence. Resource caps are interpreted by this trusted verifier.
4. Implement `upper_bounds(contract, original, proposal)` to bound every protected proposal-minus-original loss. Use finite Python floats, integers or exact fractions. Omit unavailable metrics or return `None`. Among certified improvements, selection is lexicographic in registered loss order and then branch order. Missing or positive bounds and ties retain a feasible original in the absence of a better certified branch. An unresolved original returns no executable. An affirmatively infeasible original can be replaced by a separately certified feasible policy.
5. Execute the returned `policy` only after that decision. Mutable hardware snapshots and deployment samples cannot supply independent evidence for the earlier decision.

## Compare A2 with its protected direct incumbent

`compile_direct` and `direct_diagram` return an incumbent ticket in the receipt. Pass the receipt to the corresponding bounded entry point. For recipe-based compilation, omitting it invokes `adapter.direct_branch(job)` if registered. For diagram search, omitting it computes the direct priority derivation. The `budget` and `max_candidates` arguments constrain new proposals. Record the physical shot/time quota in the task or in `HostJob.execution_budget`. A previously delivered incumbent consumes no additional proposal budget.

The trusted verifier first rechecks feasibility and the protected direct policy's bounds against B under the same task, device, execution budget and metric contract. A new policy needs a nonpositive upper difference on each protected loss against B and A1, with a negative upper difference for at least one loss against A1. Implement `upper_bounds_against(contract, baseline_assessment, candidate_assessment)` for this second comparison. An exact pairwise verifier can explicitly alias the existing `upper_bounds` method. Original-host-specific bounds cannot silently certify arbitrary reference policies. A missing pairwise bound keeps an already certified A1. An unavailable direct ticket or changed scope gives `unresolved_incumbent` with no program. If A1 retained B, a single verified comparison to B covers both references.

For immutable scalars and nested structured data (including dataclasses), the ticket snapshots the task, hardware, explicit execution budget, stage roles, full original program, metric contract and delivered direct program. Mutation invalidates the ticket. For opaque native SDK objects, provide `verifier.incumbent_key(job, direct_policy)` returning a stable digest of all these inputs and the associated physical calibration. Missing identity evidence withholds release. The same model guard precedes independent physical deployment acceptance.

A maximized reward can be declared as `ReleaseContract(model, ("time", "risk"), reward="J")`. The verifier must then supply an upper bound on `J(reference) - J(proposal)` under the key `"-J"`. When achieving the best reward is the primary objective, register `("-J", "time", "risk")` in that order. Eligible proposals are ranked lexicographically by certified differences relative to the same B. The registered order breaks exact ranking ties. An all-metric tie with A1 returns that same incumbent object.

For v0.2.0 model comparisons, choose `legacy_host_only=True` on the bounded or diagram search, or run an unchanged study with the separately archived pinned wheel. From the submission root:

```sh
python software/openquantum-vqa-harness/scripts/replay_v020.py extension/second-review/adapter/diagram_hardware.py
```

## Typed diagram search

Use named `Port` objects with a declared state or frame type. Each `Stage` carries typed inputs and outputs and a stable implementation key distinguishing its native, calibration and classical behavior. `OpenDiagram` checks linear use of wires and acyclicity. Represent every copy, discard and feedback register explicitly. A relation matcher takes `(job, graph)` and yields `Match(selected_stage_ids, replacement)`. Replacement preserves crossing wire names and types and must use exposed stage roles. The verifier checks semantic equivalence and hardware feasibility. Coverage applies to these finite, deterministic matchers and search limits. A feasible optimum beyond the supplied library remains outside that coverage.

## Executable example

For a host with several implementations of the same stages, pass
`implementation_relation(alternatives)` as a matcher to `direct_diagram` or
`search_diagram`. The callback `alternatives(job, graph, stage)` supplies
an ordered, finite collection of `Stage` objects. Each uses the same stage
ID, role and port names. Connected stages may jointly change the type of an
internal register, for example when moving state preparation, gates and
readout to a different set of physical sites. The matcher generates compatible
combinations while preserving the types at the job boundary. The verifier
then checks each complete proposal and computes its costs. Enumeration can
grow exponentially with the number of independent choices. Use a separate
graph rewrite when changing the number of stages or their connectivity.

```sh
python -m vqa_harness.examples.feedback
```

A two-round client measures a bit, prepares the next input from that result, and returns both measurement bits. A physical X is followed by an independent bit flip of probability 1/100 in the model. Each three-X block and its one-X replacement have the same ideal action. The exact answer-failure probabilities are computed for the full feedback job. The host takes six X gates, the direct one-block recipe takes four, and the bounded two-block winner takes two. Requiring an unavailable measured-chip failure bound retains the identical original executable for all four entry points. Sampling in the example follows the decision.

The source tests include the exercised adapter interfaces, a simultaneous four-stage replacement, signed Pauli readout and a search through a more expensive intermediate. In the manuscript archive, `extension/second-review/adapter` contains the task-specific Qiskit LightCone replays on archived host outputs.

## Qualifying a frozen policy for physical deployment

Model release yields a candidate. Deployment additionally requires an immutable physical scope, a registered minimum whole-job success probability and confidence, and every named physical cap. For example, named physical losses may include the candidate-minus-saved answer risk with upper cap zero. The client must certify that differential from suitable paired evidence. Freeze the complete native words, placement, pulse parameters, readout and classical controller. The following functions are importable from `vqa_harness`.

```python
from fractions import Fraction
from vqa_harness import DeploymentContract, qualify_saved, freeze_deployment, admit_deployment

contract = DeploymentContract(scope_id, Fraction(95, 100), Fraction(95, 100),
                              (("paired_physical_answer_loss", 0),))
prior = qualify_saved(contract, original_host, prior_evidence_reference, registry)
frozen = freeze_deployment(contract, selected_program, registry, fallback=prior["saved"])
# Independent physical validation occurs after the registry seal.
decision = admit_deployment(frozen, validation_evidence_reference, registry)
```

A registry supplies four trusted methods. `subject(policy)` hashes all executable and controller content. `seal(contract, candidate_digest, standby_record)` durably binds the entire physical contract and the designated standby to a strictly ordered acquisition marker. `verify_seal(contract, digest, standby_record, marker)` authenticates that record. `read_evidence(contract, policy, reference)` validates physical provenance, scope, current calibration validity, trial independence where applicable and all bound certifications, returning a verified `PhysicalEvidence` or no evidence. A deterministic certificate must verify the delivered answer for every allowed case in the physical contract. A JSON field asserting that a proof exists is insufficient.

The decision is `qualified_candidate`, `qualified_standby`, or `not_authorized`. Any observed task-gate failure, missing physical answer evidence, insufficient probability evidence or unavailable/exceeded physical bound blocks the candidate. All observed jobs must pass. In the independent Bernoulli case the exact one-sided tail at the predeclared target must also fit the error budget. The implementation splits the family-wise budget using the verified simultaneous confidence of all physical caps. At an absolute target of 0.95, with exact caps and confidence 0.95, 59/59 independent complete-job passes suffice and 58/58 do not. Finite samples cannot certify a success probability of one. Only an independently verified universal physical proof supports that contract. Technical repeats or individual shots cannot be counted as independent complete jobs without supporting evidence.

An authorized standby has its own prior qualification under the identical contract. The registry rechecks it on use. An available host with unknown physical qualification provides no fallback. Scores from another arm evaluated on the same new batch cannot supply a retrospective fallback. `audit_physical_evidence` supports read-only review of archived results and returns no executable.

For a device adapter with a submission callback, call `dispatch_qualified(frozen, validation_evidence_reference, registry, submit)`. It rechecks acceptance immediately before invoking `submit(policy)`. Rejected decisions return without calling the device hook. The current package supplies the gate and callback interface. The host supplies its authenticated device client and evidence registry.
