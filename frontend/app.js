(() => {
  const form = document.getElementById('search-form');
  const queryInput = document.getElementById('query');
  const searchButton = document.getElementById('search-button');
  const weightInput = document.getElementById('lexical-weight');
  const weightOutput = document.getElementById('weight-value');
  const limitInput = document.getElementById('result-limit');
  const resultsSection = document.getElementById('results-section');
  const resultsHeading = document.getElementById('results-heading');
  const resultsTitle = document.getElementById('results-title');
  const resultsMeta = document.getElementById('results-meta');
  const status = document.getElementById('status');
  const resultsList = document.getElementById('results-list');

  function updateWeightLabel() {
    const lexical = Number(weightInput.value);
    const semantic = 1 - lexical;
    weightOutput.textContent = `${Math.round(lexical * 100)}% lexical · ${Math.round(semantic * 100)}% semantic`;
  }

  function makeElement(tag, className, text) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (text !== undefined && text !== null) element.textContent = String(text);
    return element;
  }

  function formatTimestamp(seconds) {
    if (seconds === null || seconds === undefined || !Number.isFinite(Number(seconds))) return 'Timestamp unavailable';
    const total = Math.max(0, Math.floor(Number(seconds)));
    const hours = Math.floor(total / 3600);
    const minutes = Math.floor((total % 3600) / 60);
    const secs = total % 60;
    return [hours, minutes, secs].map(value => String(value).padStart(2, '0')).join(':');
  }

  function passageTime(passage) {
    const start = formatTimestamp(passage.start_s);
    if (passage.end_s === null || passage.end_s === undefined) return start;
    return `${start} – ${formatTimestamp(passage.end_s)}`;
  }

  function appendPassage(parent, passage, label, extraClass = '') {
    const wrapper = makeElement('div', `additional-passage ${extraClass}`.trim());
    wrapper.append(makeElement('div', 'passage-label', label));
    wrapper.append(makeElement('p', 'passage-text', passage.text || 'No passage text available.'));
    wrapper.append(makeElement('span', 'passage-time', passageTime(passage)));
    parent.append(wrapper);
  }

  function renderResults(data) {
    resultsList.replaceChildren();
    resultsHeading.hidden = false;
    resultsTitle.textContent = `${data.results.length} ${data.results.length === 1 ? 'episode' : 'episodes'}`;
    resultsMeta.textContent = Number.isFinite(data.elapsed_ms) ? `${data.elapsed_ms.toFixed(1)} ms` : '';

    if (!data.results.length) {
      status.hidden = false;
      status.className = 'status';
      status.textContent = 'No matching episodes found. Try a different query or adjust the search balance.';
      return;
    }

    status.hidden = true;
    data.results.forEach((episode, index) => {
      const card = makeElement('article', 'episode-card');
      const top = makeElement('div', 'episode-top');
      top.append(makeElement('div', 'episode-number', String(index + 1).padStart(2, '0')));
      const heading = makeElement('div', 'episode-heading');
      heading.append(makeElement('h3', 'episode-title', episode.episode_title || 'Untitled episode'));
      heading.append(makeElement('div', 'episode-id', `Episode ID · ${episode.episode_id}`));
      top.append(heading);
      card.append(top);

      appendPassage(card, episode.best_passage || {}, 'Best matching passage');
      const additional = Array.isArray(episode.additional_passages) ? episode.additional_passages : [];
      if (additional.length) {
        const details = document.createElement('details');
        details.className = 'additional';
        const summary = makeElement('summary', '', `Show ${additional.length} additional ${additional.length === 1 ? 'passage' : 'passages'}`);
        details.append(summary);
        additional.forEach((passage, passageIndex) => {
          appendPassage(details, passage, `Additional passage ${passageIndex + 1}`);
        });
        card.append(details);
      } else {
        card.append(makeElement('p', 'empty-additional', 'No other matching passages in this episode.'));
      }
      resultsList.append(card);
    });
  }

  function showError(message) {
    resultsList.replaceChildren();
    resultsHeading.hidden = true;
    status.hidden = false;
    status.className = 'status error';
    status.textContent = message;
  }

  weightInput.addEventListener('input', updateWeightLabel);

  form.addEventListener('submit', async event => {
    event.preventDefault();
    const query = queryInput.value.trim();
    const requestedLimit = Number.parseInt(limitInput.value, 10);
    if (!query) {
      queryInput.focus();
      return;
    }
    if (!Number.isInteger(requestedLimit) || requestedLimit < 1 || requestedLimit > 25) {
      showError('Choose an episode count between 1 and 25.');
      limitInput.focus();
      return;
    }

    searchButton.disabled = true;
    searchButton.textContent = 'Searching…';
    resultsSection.setAttribute('aria-busy', 'true');
    resultsList.replaceChildren();
    resultsHeading.hidden = true;
    status.hidden = false;
    status.className = 'status';
    status.textContent = 'Searching the transcript archive…';

    try {
      const response = await fetch('/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          lexical_weight: Number(weightInput.value),
          limit: requestedLimit
        })
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        const detail = typeof payload.detail === 'string' ? payload.detail : `Search request failed (${response.status}).`;
        throw new Error(detail);
      }
      renderResults(payload);
    } catch (error) {
      showError(error instanceof Error ? error.message : 'An unexpected error occurred while searching.');
    } finally {
      searchButton.disabled = false;
      searchButton.innerHTML = 'Search <span aria-hidden="true">↗</span>';
      resultsSection.setAttribute('aria-busy', 'false');
    }
  });

  updateWeightLabel();
})();
