/**
 * Data Explorer Application
 * Main entry point - initializes all modules
 */

import { state, subscribe } from './state.js';
import { initDuckDB } from './duckdb.js';
import { runQuery } from './query-builder.js';
import { initUI, renderTableList, renderColumnPanel, renderGrid, renderSQL, updateStatus, showLoading } from './ui.js';

/**
 * Initialize the application
 */
async function init() {
    console.log('Initializing Data Explorer...');

    // Subscribe to state changes
    subscribe(handleStateChange);

    // Initialize UI event listeners
    initUI();

    // Initialize DuckDB
    try {
        showLoading(true);
        await initDuckDB();
        updateStatus('ready', 'Ready - Load a database');
        console.log('Data Explorer ready');
    } catch (err) {
        console.error('Failed to initialize DuckDB:', err);
        updateStatus('error', 'Failed to initialize DuckDB');
    }
    showLoading(false);
}

/**
 * Handle state changes and update UI
 */
function handleStateChange(changeType) {
    console.log('State change:', changeType);

    switch (changeType) {
        case 'database-loaded':
            renderTableList();
            renderColumnPanel();
            break;

        case 'base-table-changed':
            renderTableList();
            renderColumnPanel();
            renderGrid();
            renderSQL();
            break;

        case 'columns-changed':
            // Run query when columns change
            runQuery();
            break;

        case 'query-complete':
        case 'query-error':
        case 'sort-changed':
        case 'filter-changed':
        case 'page-changed':
            renderGrid();
            renderSQL();
            break;
    }
}

// Start the application
init();
