(function () {
    'use strict';

    const results = window.CARBenchResults;
    const {
        data,
        escapeHtml,
        formatCost,
        formatInteger,
        formatLatency,
        formatPercent,
        formatRank,
        formatSeconds,
        teamsForTrack,
        winnerTeams,
    } = results;

    const viewButtons = [...document.querySelectorAll('[data-result-view]')];
    const viewPanels = [...document.querySelectorAll('[data-result-panel]')];
    const trackButtons = [...document.querySelectorAll('[data-leaderboard-track]')];
    const leaderboardBody = document.getElementById('leaderboard-body');
    const detailToggle = document.getElementById('detail-toggle');
    const sortButtons = [...document.querySelectorAll('[data-sort-key]')];
    const requestedTeam = new URLSearchParams(location.search).get('team');
    const baselines = window.CAR_BENCH_BASELINES ?? [];

    let activeView = location.hash === '#explore' || requestedTeam ? 'explore' : 'leaderboard';
    let activeTrack = location.hash === '#track-2' ? 'track_2' : 'track_1';
    let leaderboardSortKey = 'pass3';
    let leaderboardSortDirection = 'desc';
    let selectedKey = data.teams.some(team => team.key === requestedTeam)
        ? requestedTeam
        : 'track_1__team-35';

    function resultUrl(teamKey) {
        return `leaderboard.html?team=${encodeURIComponent(teamKey)}#explore`;
    }

    function reportDownloadName(teamName) {
        const slug = teamName.trim().toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '');
        return `${slug || 'submission'}-technical-report.pdf`;
    }

    const reportRevisions = {
        'track_1__team-7': '20260808',
    };

    function reportUrl(team) {
        const revision = reportRevisions[team.key];
        return revision ? `${team.report}?v=${revision}` : team.report;
    }

    function selectView(view, updateUrl = true) {
        activeView = view;
        viewButtons.forEach(button => {
            const active = button.dataset.resultView === view;
            button.classList.toggle('active', active);
            button.setAttribute('aria-selected', String(active));
        });
        viewPanels.forEach(panel => panel.classList.toggle('active', panel.dataset.resultPanel === view));

        if (updateUrl) {
            const suffix = view === 'explore'
                ? `?team=${encodeURIComponent(selectedKey)}#explore`
                : (activeTrack === 'track_2' ? '#track-2' : '');
            history.replaceState(null, '', `leaderboard.html${suffix}`);
        }
    }

    viewButtons.forEach(button => button.addEventListener('click', () => selectView(button.dataset.resultView)));

    function detailCell(label, value, note = '') {
        return `<td class="num detail-column" data-label="${escapeHtml(label)}">
            ${escapeHtml(value)}${note ? `<small>${escapeHtml(note)}</small>` : ''}
        </td>`;
    }

    function metricCell(label, value, note = '') {
        return `<td class="num" data-label="${escapeHtml(label)}">
            ${escapeHtml(value)}${note ? `<small>${escapeHtml(note)}</small>` : ''}
        </td>`;
    }

    function attachExploreLinks(container) {
        container.querySelectorAll('[data-explore-team]').forEach(link => {
            link.addEventListener('click', event => {
                event.preventDefault();
                selectTeam(link.dataset.exploreTeam);
                selectView('explore');
                document.getElementById('view-explore').scrollIntoView({ behavior: 'smooth', block: 'start' });
            });
        });
    }

    function updateSortHeaders() {
        document.querySelectorAll('[data-sort-header]').forEach(header => {
            const active = header.dataset.sortHeader === leaderboardSortKey;
            header.setAttribute('aria-sort', active
                ? (leaderboardSortDirection === 'asc' ? 'ascending' : 'descending')
                : 'none');
            const indicator = header.querySelector('.sort-indicator');
            if (indicator) indicator.textContent = active
                ? (leaderboardSortDirection === 'asc' ? '↑' : '↓')
                : '↕';
        });
    }

    function sortedLeaderboardTeams() {
        const trackBaselines = baselines.filter(team => team.track === activeTrack);
        return [...teamsForTrack(activeTrack), ...trackBaselines].sort((left, right) => {
            const leftValue = left.metrics[leaderboardSortKey];
            const rightValue = right.metrics[leaderboardSortKey];
            const leftAvailable = Number.isFinite(leftValue);
            const rightAvailable = Number.isFinite(rightValue);
            if (!leftAvailable && !rightAvailable) return left.teamName.localeCompare(right.teamName);
            if (!leftAvailable) return 1;
            if (!rightAvailable) return -1;
            if (leftValue !== rightValue) {
                return leaderboardSortDirection === 'asc'
                    ? leftValue - rightValue
                    : rightValue - leftValue;
            }
            if (Boolean(left.isBaseline) !== Boolean(right.isBaseline)) {
                return left.isBaseline ? 1 : -1;
            }
            const officialRankDifference = left.metrics.rank.start - right.metrics.rank.start;
            return officialRankDifference || left.teamName.localeCompare(right.teamName);
        });
    }

    function renderLeaderboard() {
        const teams = sortedLeaderboardTeams();
        const rankedSubmissionCount = teams.filter(team => !team.isBaseline).length;
        document.getElementById('leaderboard-track-title').textContent = data.tracks[activeTrack].label.replace(' · ', ': ');
        document.getElementById('leaderboard-track-count').textContent = `${rankedSubmissionCount} ranked submissions · organizer baseline shown for reference`;
        updateSortHeaders();

        leaderboardBody.innerHTML = teams.map(team => {
            const metrics = team.metrics;
            const star = team.award
                ? `<span class="table-award-star" title="${escapeHtml(team.award.award)}" aria-label="${escapeHtml(team.award.award)}">★</span>`
                : '';
            const rowClass = [team.award ? 'award-row' : '', team.isBaseline ? 'baseline-reference-row' : ''].filter(Boolean).join(' ');
            const rank = team.isBaseline ? '—' : formatRank(metrics.rank);
            const teamNote = team.isBaseline ? 'Non-competing reference' : (team.award?.award ?? '');
            const reportCell = team.isBaseline
                ? '<td class="report-link-cell baseline-report-cell" aria-label="No participant report">—</td>'
                : `<td class="report-link-cell"><a href="${resultUrl(team.key)}" data-explore-team="${escapeHtml(team.key)}">View &rarr;</a></td>`;
            return `<tr class="${rowClass}">
                <td><strong>${escapeHtml(rank)}</strong></td>
                <td><div class="team-name-line">${star}<strong>${escapeHtml(team.teamName)}</strong></div>${teamNote ? `<small>${escapeHtml(teamNote)}</small>` : ''}</td>
                <td class="llm-column"><span class="llm-pill" title="${escapeHtml(team.mainLlm)}">${escapeHtml(team.mainLlm)}</span></td>
                <td class="num leaderboard-pass3-cell">${escapeHtml(formatPercent(metrics.pass3))}</td>
                ${detailCell('Pass@3', formatPercent(metrics.passAt3))}
                ${detailCell('Pass@1', formatPercent(metrics.pass1))}
                ${detailCell('Successful trials', `${metrics.successfulTrials}/${metrics.successfulTrialsTotal}`)}
                ${metricCell('Success consistency', formatPercent(metrics.successConsistency))}
                ${metricCell('Task latency', formatSeconds(metrics.taskLatencyMedianMs), 'median')}
                ${metricCell('Mean tokens / trial', formatInteger(metrics.meanTokensPerTrial))}
                ${metricCell('Estimated cost / trial', formatCost(metrics.estimatedCostPerTrialUsd))}
                ${reportCell}
            </tr>`;
        }).join('');
        attachExploreLinks(leaderboardBody);
    }

    function selectTrack(track, updateUrl = true) {
        activeTrack = track;
        trackButtons.forEach(button => {
            const active = button.dataset.leaderboardTrack === track;
            button.classList.toggle('active', active);
            button.setAttribute('aria-selected', String(active));
        });
        renderLeaderboard();
        if (updateUrl && activeView === 'leaderboard') {
            history.replaceState(null, '', track === 'track_2' ? 'leaderboard.html#track-2' : 'leaderboard.html');
        }
    }

    trackButtons.forEach(button => button.addEventListener('click', () => selectTrack(button.dataset.leaderboardTrack)));

    sortButtons.forEach(button => button.addEventListener('click', () => {
        const key = button.dataset.sortKey;
        if (leaderboardSortKey === key) {
            leaderboardSortDirection = leaderboardSortDirection === 'asc' ? 'desc' : 'asc';
        } else {
            leaderboardSortKey = key;
            leaderboardSortDirection = button.dataset.sortDefault;
        }
        renderLeaderboard();
    }));

    detailToggle.addEventListener('click', () => {
        const expanded = document.body.classList.toggle('leaderboard-details-visible');
        detailToggle.setAttribute('aria-expanded', String(expanded));
        detailToggle.setAttribute('aria-label', expanded
            ? 'Hide Pass@3, Pass@1, and successful trials'
            : 'Show Pass@3, Pass@1, and successful trials');
        detailToggle.innerHTML = expanded
            ? 'Fewer scores <span aria-hidden="true">‹</span>'
            : 'More scores <span aria-hidden="true">›</span>';
    });

    const awardGrid = document.getElementById('award-grid');
    const awardCard = team => {
        const description = team.award.description
            ? `<p class="award-rationale">${escapeHtml(team.award.description)}</p>`
            : `<p>${escapeHtml(team.award.innovation)}</p><p class="award-rationale">${escapeHtml(team.award.rationale)}</p>`;
        return `<article class="award-result-card ${team.track === 'track_2' ? 'track-two' : ''}">
            <div class="award-kicker"><span>★ ${escapeHtml(team.award.card_award ?? team.award.award)}</span><span>${escapeHtml(data.tracks[team.track].shortLabel)}</span></div>
            <h3>${escapeHtml(team.teamName)}</h3>
            ${description}
            <div class="award-card-footer"><span><strong>${escapeHtml(formatPercent(team.metrics.pass3))}</strong> Pass³</span><a href="${resultUrl(team.key)}" data-explore-team="${escapeHtml(team.key)}">View report &rarr;</a></div>
        </article>`;
    };
    awardGrid.innerHTML = ['track_1', 'track_2'].map(track => {
        const [trackNumber, trackName] = data.tracks[track].label.split(' · ');
        const cards = winnerTeams().filter(team => team.track === track).map(awardCard).join('');
        return `<section class="award-track-column ${track === 'track_2' ? 'track-two' : ''}" aria-label="${escapeHtml(trackNumber)} award winners">
            <h3 class="award-track-heading"><span>${escapeHtml(trackNumber)}</span><small>${escapeHtml(trackName)}</small></h3>
            <div class="award-track-list">${cards}</div>
        </section>`;
    }).join('');
    attachExploreLinks(awardGrid);

    const searchInput = document.getElementById('team-search');
    const trackFilter = document.getElementById('track-filter');
    const awardFilter = document.getElementById('award-filter');
    const teamList = document.getElementById('explorer-team-list');
    const emptyState = document.getElementById('team-list-empty');

    function filteredTeams() {
        const query = searchInput.value.trim().toLowerCase();
        return data.teams.filter(team => {
            const trackMatch = trackFilter.value === 'all' || team.track === trackFilter.value;
            const awardMatch = !awardFilter.checked || Boolean(team.award);
            const text = `${team.teamName} ${team.mainLlm}`.toLowerCase();
            return trackMatch && awardMatch && (!query || text.includes(query));
        });
    }

    function renderTeamList() {
        const teams = filteredTeams();
        emptyState.hidden = teams.length > 0;
        teamList.innerHTML = teams.map(team => `<button class="submission-team-row${team.key === selectedKey ? ' selected' : ''}${team.award ? ' winner' : ''}" type="button" data-team-key="${escapeHtml(team.key)}" aria-current="${team.key === selectedKey ? 'true' : 'false'}">
            <span class="submission-track-tag">${team.track === 'track_1' ? 'T1' : 'T2'}</span>
            <span class="submission-team-copy"><strong>${escapeHtml(team.teamName)}</strong><small>${escapeHtml(team.mainLlm)}</small></span>
            ${team.award ? `<span class="submission-award-star" title="${escapeHtml(team.award.award)}">★</span>` : '<span></span>'}
        </button>`).join('');
        teamList.querySelectorAll('[data-team-key]').forEach(button => button.addEventListener('click', () => selectTeam(button.dataset.teamKey)));
    }

    const metricRankExclusions = {
        meanTokensPerTrial: new Set([
            'track_1__team-6', 'track_1__team-9',
            'track_2__team-3', 'track_2__team-13', 'track_2__team-17',
            'track_2__team-27', 'track_2__team-41',
        ]),
        estimatedCostPerTrialUsd: new Set([
            'track_1__team-6', 'track_1__team-9', 'track_1__team-10',
            'track_2__team-3', 'track_2__team-13', 'track_2__team-17',
            'track_2__team-19', 'track_2__team-27', 'track_2__team-41',
        ]),
    };

    function metricRank(team, metricKey, higherIsBetter = true) {
        const excluded = metricRankExclusions[metricKey] ?? new Set();
        const value = team.metrics[metricKey];
        if (!Number.isFinite(value) || excluded.has(team.key)) return null;

        const values = teamsForTrack(team.track)
            .filter(peer => Number.isFinite(peer.metrics[metricKey]) && !excluded.has(peer.key))
            .map(peer => peer.metrics[metricKey])
            .sort((left, right) => higherIsBetter ? right - left : left - right);
        const positions = values
            .map((candidate, index) => candidate === value ? index + 1 : null)
            .filter(Boolean);
        const minimum = Math.min(...values);
        const maximum = Math.max(...values);
        const colorScore = minimum === maximum
            ? 0.5
            : (higherIsBetter ? (value - minimum) : (maximum - value)) / (maximum - minimum);
        return {
            start: positions[0],
            end: positions[positions.length - 1],
            total: values.length,
            minimum,
            maximum,
            higherIsBetter,
            colorScore,
        };
    }

    function rankLabel(rank) {
        if (!rank) return 'Unranked';
        const place = rank.start === rank.end ? rank.start : `${rank.start}–${rank.end}`;
        return `Rank ${place} of ${rank.total}`;
    }

    function categoryScoreLine(metrics, metricKey) {
        const categoryMetricKeys = {
            pass3: 'pass3',
            passAt3: 'pass_at_3',
            pass1: 'pass1',
        };
        const categoryMetricKey = categoryMetricKeys[metricKey];
        const categories = metrics.categories;
        if (!categoryMetricKey || !categories) return '';

        const labels = [
            ['base', 'Base'],
            ['hallucination', 'Hallucination'],
            ['disambiguation', 'Disambiguation'],
        ];
        const scores = labels
            .map(([key, label]) => {
                const value = categories[key]?.[categoryMetricKey];
                return Number.isFinite(value) ? `${label}: ${formatPercent(value)}` : null;
            })
            .filter(Boolean);
        return scores.length ? scores.join(' · ') : '';
    }

    function metricItem(team, metricKey, label, value, note, formatter, higherIsBetter = true, categoryScores = '') {
        const rank = metricRank(team, metricKey, higherIsBetter);
        const stateClass = rank ? ' ranked' : ' unranked';
        const colorStyle = rank ? ` style="--metric-hue:${Math.round(rank.colorScore * 120)}"` : '';
        const range = rank
            ? `${data.tracks[team.track].shortLabel} range: ${formatter(rank.minimum)}–${formatter(rank.maximum)} · ${rank.higherIsBetter ? 'higher' : 'lower'} is better`
            : 'Comparison unavailable';
        return `<div class="score-list-item${stateClass}"${colorStyle}>
            <div class="score-item-label"><span>${escapeHtml(label)}</span>${rank?.start === 1 ? '<em>Best</em>' : ''}</div>
            <div class="score-item-value"><strong>${escapeHtml(value)}</strong><small class="score-rank">${escapeHtml(rankLabel(rank))}</small></div>
            <small class="score-item-note">${escapeHtml(note)}</small>
            ${categoryScores ? `<small class="score-item-category-scores">${escapeHtml(categoryScores)}</small>` : ''}
            <small class="score-item-range">${escapeHtml(range)}</small>
        </div>`;
    }

    function renderSelectedTeam() {
        const team = data.teams.find(item => item.key === selectedKey) ?? data.teams[0];
        const metrics = team.metrics;

        document.getElementById('team-eyebrow').textContent = data.tracks[team.track].label;
        document.getElementById('team-name').textContent = team.teamName;
        document.getElementById('team-model').textContent = `LLM used: ${team.mainLlm}`;
        const awardBadge = document.getElementById('team-award-badge');
        awardBadge.hidden = !team.award;
        awardBadge.textContent = team.award ? `★ ${team.award.award}` : '';
        document.getElementById('summary-punchline').textContent = team.summary.punchline;
        document.getElementById('summary-body').textContent = team.summary.body;

        document.getElementById('primary-score').textContent = formatPercent(metrics.pass3);
        document.getElementById('primary-category-scores').textContent = categoryScoreLine(metrics, 'pass3');
        document.getElementById('primary-rank').textContent = `Rank ${formatRank(metrics.rank)} of ${metrics.rank.total}`;
        document.getElementById('score-list').innerHTML = [
            metricItem(team, 'passAt3', 'Pass@3', formatPercent(metrics.passAt3), 'Task succeeds in at least one of three trials.', formatPercent, true, categoryScoreLine(metrics, 'passAt3')),
            metricItem(team, 'pass1', 'Pass@1', formatPercent(metrics.pass1), 'Mean single-trial success across the benchmark.', formatPercent, true, categoryScoreLine(metrics, 'pass1')),
            metricItem(team, 'successfulTrials', 'Successful trials', `${metrics.successfulTrials}/${metrics.successfulTrialsTotal}`, 'Successful task-runs out of 90 total trials.', value => `${formatInteger(value)}/90`),
            metricItem(team, 'successConsistency', 'Success consistency', formatPercent(metrics.successConsistency), '1/3 succeeds = 0% · 2/3 = 50% · 3/3 = 100%.', formatPercent),
            metricItem(team, 'taskLatencyMedianMs', 'Task latency · median', formatSeconds(metrics.taskLatencyMedianMs), `Mean ${formatSeconds(metrics.taskLatencyMeanMs)} · p95 ${formatSeconds(metrics.taskLatencyP95Ms)}.`, formatSeconds, false),
            metricItem(team, 'turnLatencyMedianMs', 'A2A turn latency · median', formatLatency(metrics.turnLatencyMedianMs), 'Median latency for one agent-to-agent turn.', formatLatency, false),
            metricItem(team, 'meanTokensPerTrial', 'Mean tokens / trial', formatInteger(metrics.meanTokensPerTrial), metrics.meanTokensPerTrial == null ? 'Token accounting unavailable.' : 'Reported total tokens averaged over evaluated trials.', formatInteger, false),
            metricItem(team, 'estimatedCostPerTrialUsd', 'Estimated cost / trial', formatCost(metrics.estimatedCostPerTrialUsd), metrics.estimatedCostPerTrialUsd == null ? 'Pricing or token accounting unavailable.' : data.evaluation.costNote, formatCost, false),
        ].join('');

        const rawReport = document.getElementById('raw-report-link');
        const downloadReport = document.getElementById('download-report-link');
        const source = reportUrl(team);
        rawReport.href = source;
        downloadReport.href = source;
        downloadReport.download = reportDownloadName(team.teamName);
        const viewer = document.getElementById('report-frame');
        viewer.title = `${team.teamName} technical report`;
        viewer.src = `${source}#view=FitH&toolbar=1&navpanes=0`;
    }

    function selectTeam(key) {
        if (!data.teams.some(team => team.key === key)) return;
        selectedKey = key;
        renderTeamList();
        renderSelectedTeam();
        if (activeView === 'explore') history.replaceState(null, '', resultUrl(selectedKey));
    }

    searchInput.addEventListener('input', renderTeamList);
    trackFilter.addEventListener('change', renderTeamList);
    awardFilter.addEventListener('change', renderTeamList);

    selectTrack(activeTrack, false);
    renderTeamList();
    renderSelectedTeam();
    selectView(activeView, false);
})();
