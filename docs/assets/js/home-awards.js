(function () {
    'use strict';

    const awardGrid = document.getElementById('award-grid');
    const results = window.CARBenchResults;
    if (!awardGrid || !results) return;

    const {
        data,
        escapeHtml,
        formatPercent,
        winnerTeams,
    } = results;

    const resultUrl = teamKey => `leaderboard.html?team=${encodeURIComponent(teamKey)}#explore`;

    const awardCard = team => {
        const description = team.award.description
            ? `<p class="award-rationale">${escapeHtml(team.award.description)}</p>`
            : `<p>${escapeHtml(team.award.innovation)}</p><p class="award-rationale">${escapeHtml(team.award.rationale)}</p>`;
        const prizes = team.award.prizes?.length
            ? `<div class="award-prizes"><strong>Prizes</strong><ul>${team.award.prizes.map(prize => `<li>${escapeHtml(prize)}</li>`).join('')}</ul></div>`
            : '';

        return `<article class="award-result-card ${team.track === 'track_2' ? 'track-two' : ''}">
            <div class="award-kicker"><span>★ ${escapeHtml(team.award.card_award ?? team.award.award)}</span><span>${escapeHtml(data.tracks[team.track].shortLabel)}</span></div>
            <h3>${escapeHtml(team.teamName)}</h3>
            ${description}
            ${prizes}
            <div class="award-card-footer"><span><strong>${escapeHtml(formatPercent(team.metrics.pass3))}</strong> Pass³</span><a href="${resultUrl(team.key)}">View report &rarr;</a></div>
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
})();
