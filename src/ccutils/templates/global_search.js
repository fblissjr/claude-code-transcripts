(function() {
    var searchBox = document.getElementById('global-search-box');
    var searchInput = document.getElementById('global-search-input');
    var searchBtn = document.getElementById('global-search-btn');
    var modal = document.getElementById('global-search-modal');
    var modalInput = document.getElementById('global-modal-search-input');
    var modalSearchBtn = document.getElementById('global-modal-search-btn');
    var modalCloseBtn = document.getElementById('global-modal-close-btn');
    var searchStatus = document.getElementById('global-search-status');
    var searchResults = document.getElementById('global-search-results');

    if (!searchBox || !modal) return;

    // Check if search index is available
    if (typeof SEARCH_INDEX === 'undefined') {
        console.warn('Search index not found');
        return;
    }

    // Show search box (progressive enhancement)
    searchBox.style.display = 'flex';

    function escapeHtml(text) {
        var div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    function escapeRegex(string) {
        return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }

    function highlightText(text, query) {
        if (!query) return escapeHtml(text);
        var terms = query.toLowerCase().split(/\s+/).filter(function(t) { return t.length > 0; });
        if (terms.length === 0) return escapeHtml(text);

        var pattern = terms.map(escapeRegex).join('|');
        var regex = new RegExp('(' + pattern + ')', 'gi');
        return escapeHtml(text).replace(regex, '<mark>$1</mark>');
    }

    function getSnippetWithContext(content, query, maxLength) {
        maxLength = maxLength || 200;
        if (content.length <= maxLength) return content;

        var pos = content.toLowerCase().indexOf(query.toLowerCase());
        if (pos === -1) {
            // Try first word of query
            var firstWord = query.split(/\s+/)[0];
            pos = content.toLowerCase().indexOf(firstWord.toLowerCase());
        }

        if (pos !== -1) {
            var start = Math.max(0, pos - maxLength / 2);
            var end = Math.min(content.length, start + maxLength);
            if (end === content.length) start = Math.max(0, end - maxLength);
            var snippet = content.substring(start, end);
            if (start > 0) snippet = '...' + snippet;
            if (end < content.length) snippet = snippet + '...';
            return snippet;
        }

        return content.substring(0, maxLength) + '...';
    }

    function formatTimestamp(timestamp) {
        if (!timestamp) return '';
        try {
            var date = new Date(timestamp);
            return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'});
        } catch (e) {
            return timestamp;
        }
    }

    function getTypeLabel(type) {
        switch (type) {
            case 'user': return 'Prompt';
            case 'assistant': return 'Response';
            case 'tool_use': return 'Tool';
            case 'tool_result': return 'Output';
            default: return type;
        }
    }

    function openModal(query) {
        modalInput.value = query || '';
        searchResults.innerHTML = '';
        searchStatus.textContent = '';
        modal.showModal();
        modalInput.focus();
        if (query) {
            performSearch(query);
        }
    }

    function closeModal() {
        modal.close();
        if (window.location.hash.startsWith('#search=')) {
            history.replaceState(null, '', window.location.pathname + window.location.search);
        }
    }

    function updateUrlHash(query) {
        if (query) {
            history.replaceState(null, '', window.location.pathname + window.location.search + '#search=' + encodeURIComponent(query));
        }
    }

    function performSearch(query) {
        if (!query.trim()) {
            searchStatus.textContent = 'Enter a search term';
            return;
        }

        updateUrlHash(query);
        searchResults.innerHTML = '';
        searchStatus.textContent = 'Searching...';

        var queryLower = query.toLowerCase();
        var terms = queryLower.split(/\s+/).filter(function(t) { return t.length > 0; });

        var results = [];
        var documents = SEARCH_INDEX.documents || [];

        for (var i = 0; i < documents.length; i++) {
            var doc = documents[i];
            var contentLower = (doc.content || '').toLowerCase();

            // Check if all terms are present
            var allTermsMatch = terms.every(function(term) {
                return contentLower.indexOf(term) !== -1;
            });

            if (allTermsMatch) {
                results.push(doc);
            }
        }

        searchStatus.textContent = 'Found ' + results.length + ' result(s)';

        // Display results (limit to 100 for performance)
        var displayLimit = Math.min(results.length, 100);
        for (var j = 0; j < displayLimit; j++) {
            var result = results[j];
            var snippet = getSnippetWithContext(result.content, query, 200);
            var highlightedSnippet = highlightText(snippet, query);

            var link = result.project + '/' + result.session + '/' + result.page;
            if (result.anchor) {
                link += '#' + result.anchor;
            }

            var resultDiv = document.createElement('div');
            resultDiv.className = 'search-result';
            resultDiv.innerHTML =
                '<a href="' + escapeHtml(link) + '">' +
                    '<div class="search-result-meta">' +
                        '<span class="search-result-project">' + escapeHtml(result.project) + '</span>' +
                        '<span class="search-result-type">' + escapeHtml(getTypeLabel(result.type)) + '</span>' +
                        '<time>' + escapeHtml(formatTimestamp(result.timestamp)) + '</time>' +
                    '</div>' +
                    '<div class="search-result-snippet">' + highlightedSnippet + '</div>' +
                '</a>';
            searchResults.appendChild(resultDiv);
        }

        if (results.length > displayLimit) {
            var moreDiv = document.createElement('div');
            moreDiv.className = 'search-more';
            moreDiv.textContent = '... and ' + (results.length - displayLimit) + ' more results';
            searchResults.appendChild(moreDiv);
        }
    }

    // Event listeners
    searchBtn.addEventListener('click', function() {
        openModal(searchInput.value);
    });

    searchInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter') {
            openModal(searchInput.value);
        }
    });

    modalSearchBtn.addEventListener('click', function() {
        performSearch(modalInput.value);
    });

    modalInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter') {
            performSearch(modalInput.value);
        }
    });

    modalCloseBtn.addEventListener('click', closeModal);

    modal.addEventListener('click', function(e) {
        if (e.target === modal) {
            closeModal();
        }
    });

    // Check for #search= in URL on page load
    if (window.location.hash.startsWith('#search=')) {
        var urlQuery = decodeURIComponent(window.location.hash.substring(8));
        if (urlQuery) {
            searchInput.value = urlQuery;
            openModal(urlQuery);
        }
    }
})();
