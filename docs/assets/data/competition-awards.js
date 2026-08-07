(() => {
    const teams = window.CAR_BENCH_RESULTS?.teams;
    if (!teams) return;

    const awards = {
        'track_1__team-35': {
            award: 'Rank Award + Innovation Award',
            card_award: 'Rank Award + ★ Innovation Award',
            award_order: 1,
            description: 'Darwin Agent’s TRACE framework separates online action from offline self-improvement. For each user turn, an Actor retrieves focused, operation-level skills, while a Curator groups completed trajectories by the skills used and contrasts successes with failures to refine existing guidance or discover missing competencies. The Curator distinguishes runtime-visible evidence from evaluator-only information and applies a de-hardcoding rule, preserving reusable behavior rather than memorized task answers. The resulting modular skill bank improved two different model backbones in the team’s experiments. On the hidden evaluation, Darwin Agent led Track 1 with 70.0% Pass³, the strongest success consistency, and perfect performance on hallucination tasks.',
        },
        'track_1__team-8': {
            award: 'Innovation Award',
            award_order: 3,
            description: '10cars uses a staged architecture that directs additional model compute to requests with unresolved values or competing interpretations. An intent contract, informed by compiled policy rules and a ledger of successful reads and actions, determines which path each request follows. Clear requests use the normal single-model path. Ambiguous requests trigger several candidate plans, which a lightweight simulator evaluates and ranks. The chosen action from either path then passes through deterministic checks for valid tool use, required information, confirmation, repetition, preferences, and follow-up obligations before execution. Local ablations showed that selective planning worked better than applying it on every turn. The architecture remained effective after a major model change and achieved 60.0% hidden Pass³ with strong consistency.',
        },
        'track_2__team-4': {
            award: 'Cerebras Innovation Award',
            award_order: 4,
            description: 'Proxima’s coroutine-bridge harness lets one model-generated Python program span dependent tool round-trips, while deterministic policy logic in guarded tool wrappers enforces compliance without extra reasoning calls. Using Cerebras gpt-oss-120b, a substantially smaller model, it matched the 60.0% hidden Pass³ achieved by several frontier-model agents in Track 1. Compared with the baseline, it significantly improved task success and consistency while reducing latency and token use. The unchanged harness reproduced the 60.0% score with GPT-5.5, strong evidence that the gains came from the harness design rather than model capacity.',
        },
        'track_1__team-25': {
            award: 'Innovation Award',
            award_order: 4,
            description: 'shiina18 offers a compelling blueprint for scalable real-world agents: strong task coverage, low operating cost, and a runtime that prefers transparent limitations over fabricated success. Its “ground, then act” pipeline withholds each proposed tool call until available capabilities, arguments, user preferences, confirmations, and relevant vehicle policies have been checked. Missing tools, parameters, or result fields are surfaced honestly to the user, while narrowly scoped model judgments handle semantic cases that deterministic code cannot resolve. On the hidden set, its cost-efficient DeepSeek V4 Flash agent achieved 53.3% Pass³ and tied for the third-highest Pass@3 in Track 1. It recorded the second-lowest estimated cost and fourth-lowest reported token use.',
        },
        'track_2__team-36': {
            award: 'Cerebras Innovation Award',
            award_order: 5,
            description: 'FreudeDrive turns Cerebras’ fast inference into selective concurrency. A zero-inference classifier decides whether each turn needs one candidate or a parallel panel with deliberately different roles, including policy-first, tool-first, and hallucination or disambiguation-focused reasoning. Canonical voting and deterministic checks over tool schemas, state, provenance, policy, and confirmation accept clean agreement immediately. Only genuine conflicts trigger bounded adjudication and repair. This design achieved 50.0% hidden Pass³ and improved every task category over the same-model baseline, using substantially more parallel reasoning with only a moderate increase in end-to-end latency.',
        },
    };

    teams.forEach(team => {
        if (team.key === 'track_2__team-33') team.award = null;
        if (team.key === 'track_1__team-37') team.award = null;
        if (awards[team.key]) team.award = awards[team.key];
    });
})();
