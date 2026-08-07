(function () {
    'use strict';

    const data = window.CAR_BENCH_RESULTS;
    if (!data) throw new Error('Competition result data is unavailable.');

    const formatPercent = value => value == null ? '—' : `${(value * 100).toFixed(2)}%`;
    const formatSeconds = value => value == null ? '—' : `${(value / 1000).toFixed(2)} s`;
    const formatLatency = value => {
        if (value == null) return '—';
        return value < 1000 ? `${value.toFixed(1)} ms` : formatSeconds(value);
    };
    const formatInteger = value => value == null
        ? '—'
        : new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 }).format(value);
    const formatCost = value => {
        if (value == null) return '—';
        return `$${value.toFixed(value < 0.1 ? 4 : 2)}`;
    };
    const formatRank = rank => rank.start === rank.end
        ? String(rank.start)
        : `${rank.start}–${rank.end}`;
    const escapeHtml = value => String(value)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#039;');
    const labelForTrack = track => data.tracks[track]?.label ?? track;
    const teamsForTrack = track => data.teams.filter(team => team.track === track);
    const winnerTeams = () => data.teams
        .filter(team => team.award)
        .sort((a, b) => a.award.award_order - b.award.award_order);

    window.CARBenchResults = {
        data,
        escapeHtml,
        formatCost,
        formatInteger,
        formatLatency,
        formatPercent,
        formatRank,
        formatSeconds,
        labelForTrack,
        teamsForTrack,
        winnerTeams,
    };
})();
