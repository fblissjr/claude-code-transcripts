/**
 * UI Rendering Functions
 * Handles all DOM manipulation and UI updates
 */

import { state, TYPE_ICONS, ARROWS, getTableDisplayName, formatValue } from './state.js';
import { isColumnSelected, addColumn, removeColumn, setBaseTable, toggleSort, setFilter, clearFilters, setPage, runQuery, formatSQLParts } from './query-builder.js';
import { getDistinctValues } from './duckdb.js';

// Current filter state
let currentFilterColumn = null;
let filterValues = [];
let selectedFilterValues = new Set();

/**
 * Initialize UI event listeners
 */
export function initUI() {
    // File input
    document.getElementById('db-file-input').addEventListener('change', handleFileSelect);

    // Table search
    document.getElementById('table-search').addEventListener('input', renderTableList);

    // Pagination
    document.getElementById('prev-page').addEventListener('click', () => {
        setPage(state.page - 1);
        runQuery();
    });
    document.getElementById('next-page').addEventListener('click', () => {
        setPage(state.page + 1);
        runQuery();
    });

    // Clear filters
    document.getElementById('clear-filters-btn').addEventListener('click', () => {
        clearFilters();
        runQuery();
    });

    // Filter dropdown
    document.getElementById('filter-search').addEventListener('input', renderFilterValues);
    document.getElementById('filter-apply').addEventListener('click', applyFilter);
    document.getElementById('filter-clear').addEventListener('click', clearCurrentFilter);

    // Close filter dropdown when clicking outside
    document.addEventListener('click', (e) => {
        const dropdown = document.getElementById('filter-dropdown');
        if (!dropdown.contains(e.target) && !e.target.closest('.col-header')) {
            hideFilterDropdown();
        }
    });

    // SQL panel toggle
    document.getElementById('sql-toggle').addEventListener('click', toggleSQLPanel);

    // Run query button
    document.getElementById('run-query-btn').addEventListener('click', () => runQuery());
}

/**
 * Handle file selection
 */
async function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        showLoading(true);
        try {
            const { loadDatabase } = await import('./duckdb.js');
            await loadDatabase(file);
            updateStatus('connected', 'Connected: ' + file.name);
        } catch (err) {
            console.error('Error loading database:', err);
            updateStatus('error', 'Error loading database');
        }
        showLoading(false);
    }
}

/**
 * Render the table list sidebar
 */
export function renderTableList() {
    const container = document.getElementById('table-list');
    const searchTerm = document.getElementById('table-search').value.toLowerCase();

    const filteredTables = state.tables.filter(t =>
        t.toLowerCase().includes(searchTerm)
    );

    container.replaceChildren();

    if (filteredTables.length === 0) {
        const emptyDiv = document.createElement('div');
        emptyDiv.className = 'text-sm text-gray-400 text-center py-4';
        emptyDiv.textContent = 'No tables found';
        container.appendChild(emptyDiv);
        return;
    }

    // Group tables
    const semanticViews = filteredTables.filter(t => t.startsWith('semantic_'));
    const facts = filteredTables.filter(t => t.startsWith('fact_'));
    const dims = filteredTables.filter(t => t.startsWith('dim_'));
    const others = filteredTables.filter(t =>
        !t.startsWith('fact_') && !t.startsWith('dim_') &&
        !t.startsWith('semantic_') && !t.startsWith('meta_') && !t.startsWith('stg_')
    );

    // Semantic views first (recommended for exploration)
    if (semanticViews.length > 0) {
        container.appendChild(createSectionHeader('Semantic Views (Recommended)'));
        semanticViews.forEach(t => container.appendChild(createSemanticViewItem(t)));
    }

    // Fact tables
    if (facts.length > 0) {
        container.appendChild(createSectionHeader('Fact Tables'));
        facts.forEach(t => container.appendChild(createTableItem(t, 'fact')));
    }

    // Dimension tables
    if (dims.length > 0) {
        container.appendChild(createSectionHeader('Dimension Tables'));
        dims.forEach(t => container.appendChild(createTableItem(t, 'dim')));
    }

    // Others (excluding staging, meta tables)
    if (others.length > 0) {
        container.appendChild(createSectionHeader('Other'));
        others.forEach(t => container.appendChild(createTableItem(t, 'other')));
    }
}

function createSectionHeader(text) {
    const header = document.createElement('div');
    header.className = 'text-xs text-gray-500 uppercase px-2 py-1 mt-3 first:mt-0 font-medium';
    header.textContent = text;
    return header;
}

function createSemanticViewItem(tableName) {
    const isActive = tableName === state.baseTable;
    const displayName = getTableDisplayName(tableName);

    const div = document.createElement('div');
    div.className = `table-item semantic ${isActive ? 'active' : ''} px-3 py-2 rounded cursor-pointer text-sm`;

    const wrapper = document.createElement('div');
    wrapper.className = 'flex items-center gap-2';

    const icon = document.createElement('span');
    icon.className = 'text-indigo-500';
    icon.textContent = '\u2726'; // Star icon

    const nameDiv = document.createElement('div');
    nameDiv.className = 'font-medium text-gray-900';
    nameDiv.textContent = displayName;

    wrapper.appendChild(icon);
    wrapper.appendChild(nameDiv);
    div.appendChild(wrapper);

    const descDiv = document.createElement('div');
    descDiv.className = 'text-xs text-gray-500 ml-5';
    descDiv.textContent = getSemanticViewDescription(tableName);
    div.appendChild(descDiv);

    div.addEventListener('click', () => {
        setBaseTable(tableName);
        renderTableList();
        renderColumnPanel();
    });

    return div;
}

function getSemanticViewDescription(tableName) {
    const descriptions = {
        'semantic_sessions': 'Sessions with project info and metrics',
        'semantic_messages': 'Messages with type, model, and context',
        'semantic_tool_calls': 'Tool usage with categories and context',
        'semantic_file_operations': 'File operations by type and path'
    };
    return descriptions[tableName] || 'Pre-joined view';
}

function createTableItem(tableName, type) {
    const isActive = tableName === state.baseTable;
    const displayName = getTableDisplayName(tableName);

    const div = document.createElement('div');
    div.className = `table-item ${type} ${isActive ? 'active' : ''} px-3 py-2 rounded cursor-pointer text-sm`;

    const nameDiv = document.createElement('div');
    nameDiv.className = 'font-medium text-gray-900';
    nameDiv.textContent = displayName;

    const techDiv = document.createElement('div');
    techDiv.className = 'text-xs text-gray-500';
    techDiv.textContent = tableName;

    div.appendChild(nameDiv);
    div.appendChild(techDiv);

    div.addEventListener('click', () => {
        setBaseTable(tableName);
        renderTableList();
        renderColumnPanel();
    });

    return div;
}


function createColumnCheckbox(tableName, col) {
    const isSelected = isColumnSelected(tableName, col.name);
    const typeInfo = TYPE_ICONS[col.dataType] || TYPE_ICONS.varchar;

    const label = document.createElement('label');
    label.className = 'flex items-center gap-2 px-2 py-1 rounded hover:bg-gray-100 cursor-pointer text-sm';

    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.checked = isSelected;
    checkbox.className = 'rounded border-gray-300 text-indigo-600 focus:ring-indigo-500';

    checkbox.addEventListener('change', () => {
        if (checkbox.checked) {
            addColumn(tableName, col.name);
        } else {
            removeColumn(tableName, col.name);
        }
        runQuery();
        renderColumnPanel();
        renderTableList();
    });

    const typeSpan = document.createElement('span');
    typeSpan.className = 'type-icon ' + typeInfo.class;
    typeSpan.textContent = typeInfo.icon;

    const nameSpan = document.createElement('span');
    nameSpan.className = 'text-gray-700 truncate';
    nameSpan.textContent = col.displayName;

    label.appendChild(checkbox);
    label.appendChild(typeSpan);
    label.appendChild(nameSpan);

    return label;
}

/**
 * Render the selected columns panel
 */
export function renderColumnPanel() {
    const container = document.getElementById('column-list');
    const panel = document.getElementById('column-panel');

    if (!state.baseTable) {
        panel.classList.add('hidden');
        return;
    }

    panel.classList.remove('hidden');
    container.replaceChildren();

    // Show base table columns first
    const baseHeader = document.createElement('div');
    baseHeader.className = 'text-xs text-gray-500 uppercase px-2 py-1 font-medium';
    baseHeader.textContent = getTableDisplayName(state.baseTable);
    container.appendChild(baseHeader);

    const baseCols = state.columnMeta[state.baseTable] || [];
    baseCols.filter(c => c.isVisible).forEach(col => {
        container.appendChild(createColumnCheckbox(state.baseTable, col));
    });
}

/**
 * Render the data grid
 */
export function renderGrid() {
    const grid = document.getElementById('data-grid');
    const header = document.getElementById('grid-header');
    const body = document.getElementById('grid-body');
    const emptyState = document.getElementById('empty-state');
    const toolbar = document.getElementById('toolbar');

    if (state.selectedColumns.length === 0 || state.data.length === 0) {
        grid.classList.add('hidden');
        emptyState.classList.remove('hidden');
        toolbar.classList.add('hidden');
        return;
    }

    grid.classList.remove('hidden');
    emptyState.classList.add('hidden');
    toolbar.classList.remove('hidden');

    // Update table name and row count
    document.getElementById('table-name').textContent = getTableDisplayName(state.baseTable);
    document.getElementById('row-count').textContent = `${state.totalRows.toLocaleString()} rows`;

    // Render header
    header.replaceChildren();
    const headerRow = document.createElement('tr');

    for (const col of state.selectedColumns) {
        const colMeta = state.columnMeta[col.table]?.find(c => c.name === col.column);
        const typeInfo = TYPE_ICONS[colMeta?.dataType] || TYPE_ICONS.varchar;
        const sort = state.sorts.find(s => s.column === (col.alias || col.column));
        const hasFilter = state.filters[col.alias || col.column]?.length > 0;

        const th = document.createElement('th');
        th.className = `col-header ${sort ? 'sorted' : ''} ${hasFilter ? 'filtered' : ''}`;
        th.dataset.column = col.alias || col.column;
        th.dataset.table = col.table;

        const wrapper = document.createElement('div');
        wrapper.className = 'flex items-center gap-2';

        const typeSpan = document.createElement('span');
        typeSpan.className = 'type-icon ' + typeInfo.class;
        typeSpan.textContent = typeInfo.icon;

        const nameSpan = document.createElement('span');
        nameSpan.className = 'flex-1';
        nameSpan.textContent = colMeta?.displayName || col.column;

        // Table indicator if from dimension
        if (col.table !== state.baseTable) {
            const tableSpan = document.createElement('span');
            tableSpan.className = 'text-xs text-gray-400';
            tableSpan.textContent = `(${getTableDisplayName(col.table)})`;
            nameSpan.appendChild(document.createTextNode(' '));
            nameSpan.appendChild(tableSpan);
        }

        const sortSpan = document.createElement('span');
        sortSpan.className = 'text-gray-400';
        sortSpan.textContent = sort ? (sort.dir === 'asc' ? ` ${ARROWS.UP}` : ` ${ARROWS.DOWN_SORT}`) : '';

        wrapper.appendChild(typeSpan);
        wrapper.appendChild(nameSpan);
        wrapper.appendChild(sortSpan);

        if (hasFilter) {
            const filterSpan = document.createElement('span');
            filterSpan.className = 'text-amber-500 text-xs';
            filterSpan.textContent = '*';
            wrapper.appendChild(filterSpan);
        }

        th.appendChild(wrapper);
        th.addEventListener('click', (e) => handleColumnClick(col, e));
        headerRow.appendChild(th);
    }
    header.appendChild(headerRow);

    // Render body
    body.replaceChildren();
    for (const row of state.data) {
        const tr = document.createElement('tr');
        for (const col of state.selectedColumns) {
            const td = document.createElement('td');
            td.className = 'px-3 py-2 text-gray-900';

            const val = row[col.alias || col.column];
            const colMeta = state.columnMeta[col.table]?.find(c => c.name === col.column);
            const dataType = colMeta?.dataType || 'varchar';

            // Use type-aware formatting
            const formatted = formatValue(val, dataType);

            if (formatted.isNull || formatted.isJson) {
                const span = document.createElement('span');
                span.className = 'text-gray-400';
                span.textContent = formatted.display;
                td.appendChild(span);
            } else {
                let displayVal = formatted.display;
                if (displayVal.length > 100) {
                    displayVal = displayVal.substring(0, 100) + '...';
                }
                td.textContent = displayVal;
            }
            tr.appendChild(td);
        }
        body.appendChild(tr);
    }

    updatePagination();
    updateClearFiltersButton();
}

function handleColumnClick(col, event) {
    if (event.shiftKey) {
        showFilterDropdown(col, event.target.closest('.col-header'));
    } else {
        toggleSort(col.alias || col.column);
        runQuery();
    }
}

/**
 * Render the SQL panel using safe DOM methods
 */
export function renderSQL() {
    const sqlContent = document.getElementById('sql-content');
    sqlContent.replaceChildren();

    if (!state.lastQuery) {
        sqlContent.textContent = '-- Select columns to generate SQL';
        return;
    }

    // Parse SQL into parts and render with syntax highlighting
    const parts = formatSQLParts(state.lastQuery);
    for (const part of parts) {
        const span = document.createElement('span');
        span.className = part.class || '';
        span.textContent = part.text;
        sqlContent.appendChild(span);
    }
}

function toggleSQLPanel() {
    const panel = document.getElementById('sql-panel');
    const toggle = document.getElementById('sql-toggle');
    panel.classList.toggle('hidden');
    toggle.textContent = panel.classList.contains('hidden') ? 'Show SQL' : 'Hide SQL';
}

/**
 * Filter dropdown handling
 */
async function showFilterDropdown(col, headerEl) {
    currentFilterColumn = col;
    selectedFilterValues = new Set(state.filters[col.alias || col.column] || []);

    const dropdown = document.getElementById('filter-dropdown');
    const searchInput = document.getElementById('filter-search');

    const rect = headerEl.getBoundingClientRect();
    dropdown.style.top = (rect.bottom + 4) + 'px';
    dropdown.style.left = rect.left + 'px';
    dropdown.classList.remove('hidden');

    searchInput.value = '';
    searchInput.focus();

    filterValues = await getDistinctValues(col.table, col.column);
    renderFilterValues();
}

function renderFilterValues() {
    const container = document.getElementById('filter-values');
    const searchTerm = document.getElementById('filter-search').value.toLowerCase();

    const filtered = filterValues.filter(v =>
        String(v).toLowerCase().includes(searchTerm)
    );

    container.replaceChildren();
    for (const val of filtered.slice(0, 50)) {
        const isSelected = selectedFilterValues.has(val);
        const displayVal = val === null ? 'null' : String(val);

        const label = document.createElement('label');
        label.className = 'flex items-center gap-2 px-3 py-1 hover:bg-gray-100 cursor-pointer';

        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.checked = isSelected;
        checkbox.className = 'rounded border-gray-300 text-indigo-600';

        checkbox.addEventListener('change', () => {
            if (checkbox.checked) {
                selectedFilterValues.add(val);
            } else {
                selectedFilterValues.delete(val);
            }
        });

        const span = document.createElement('span');
        span.className = 'text-sm text-gray-700 truncate';
        span.textContent = displayVal;

        label.appendChild(checkbox);
        label.appendChild(span);
        container.appendChild(label);
    }
}

function applyFilter() {
    if (currentFilterColumn) {
        setFilter(currentFilterColumn.alias || currentFilterColumn.column, Array.from(selectedFilterValues));
        runQuery();
    }
    hideFilterDropdown();
}

function clearCurrentFilter() {
    if (currentFilterColumn) {
        setFilter(currentFilterColumn.alias || currentFilterColumn.column, []);
        runQuery();
    }
    hideFilterDropdown();
}

function hideFilterDropdown() {
    document.getElementById('filter-dropdown').classList.add('hidden');
    currentFilterColumn = null;
}

/**
 * UI helpers
 */
function updatePagination() {
    const totalPages = Math.ceil(state.totalRows / state.pageSize) || 1;
    document.getElementById('page-info').textContent = `${state.page + 1} of ${totalPages}`;
    document.getElementById('prev-page').disabled = state.page === 0;
    document.getElementById('next-page').disabled = state.page >= totalPages - 1;
}

function updateClearFiltersButton() {
    const hasFilters = Object.values(state.filters).some(v => v?.length > 0);
    document.getElementById('clear-filters-btn').classList.toggle('hidden', !hasFilters);
}

export function updateStatus(status, text) {
    const el = document.getElementById('db-status');
    el.replaceChildren();

    const dot = document.createElement('span');
    dot.className = `w-2 h-2 rounded-full ${status === 'connected' ? 'bg-green-500' : status === 'error' ? 'bg-red-500' : 'bg-gray-300'}`;

    const textSpan = document.createElement('span');
    textSpan.textContent = text;

    el.appendChild(dot);
    el.appendChild(textSpan);
}

export function showLoading(show) {
    document.getElementById('loading-overlay').classList.toggle('hidden', !show);
}
