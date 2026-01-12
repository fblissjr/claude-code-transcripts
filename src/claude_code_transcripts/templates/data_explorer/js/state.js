/**
 * State Management for Data Explorer
 * Centralized application state with change notifications
 */

// Application state
export const state = {
    // Database connection
    db: null,
    conn: null,
    dbPrefix: '',

    // Semantic model
    semanticModel: [],
    relationships: {},  // { factTable: [{ dimTable, factCol, dimCol }] }

    // Tables
    tables: [],
    factTables: [],
    dimTables: [],

    // Current selection
    baseTable: null,        // The fact table we're querying from
    selectedColumns: [],    // [{ table, column, alias }]
    joinedTables: new Set(), // Dimension tables currently joined

    // Query state
    filters: {},            // { columnAlias: [values] }
    sorts: [],              // [{ column, dir: 'asc'|'desc' }]
    page: 0,
    pageSize: 100,
    totalRows: 0,

    // Results
    lastQuery: '',          // The generated SQL
    data: [],               // Current page of results
    summary: {},            // Aggregated summary

    // Column metadata (from semantic model + schema)
    columnMeta: {},         // { tableName: [{ name, type, displayName, ... }] }
};

// Unicode arrow constants for UI
export const ARROWS = {
    RIGHT: '\u25B6',
    DOWN: '\u25BC',
    UP: '\u2191',
    DOWN_SORT: '\u2193'
};

// Type icon mapping
export const TYPE_ICONS = {
    integer:   { icon: '123', class: 'type-integer' },
    bigint:    { icon: '123', class: 'type-integer' },
    float:     { icon: '1.2', class: 'type-float' },
    double:    { icon: '1.2', class: 'type-float' },
    decimal:   { icon: '1.2', class: 'type-float' },
    varchar:   { icon: 'Abc', class: 'type-varchar' },
    text:      { icon: 'Abc', class: 'type-varchar' },
    timestamp: { icon: 'cal', class: 'type-timestamp' },
    date:      { icon: 'cal', class: 'type-date' },
    boolean:   { icon: 'T/F', class: 'type-boolean' },
    json:      { icon: '{ }', class: 'type-json' },
};

// State change listeners
const listeners = new Set();

export function subscribe(callback) {
    listeners.add(callback);
    return () => listeners.delete(callback);
}

export function notify(changeType) {
    listeners.forEach(cb => cb(changeType));
}

// Helper to check if a table is a fact table
export function isFactTable(tableName) {
    return tableName.startsWith('fact_');
}

// Helper to check if a table is a dimension table
export function isDimTable(tableName) {
    return tableName.startsWith('dim_');
}

// Get relationships for a fact table
export function getRelationships(factTable) {
    return state.relationships[factTable] || [];
}

// Get display name for a table
export function getTableDisplayName(tableName) {
    const meta = state.semanticModel.find(m => m.tableName === tableName);
    if (meta?.tableDisplayName) return meta.tableDisplayName;

    // Generate from table name
    return tableName
        .replace(/^(dim_|fact_|stg_|meta_)/, '')
        .replace(/_/g, ' ')
        .replace(/\b\w/g, c => c.toUpperCase());
}

// Get column metadata for a table
export function getColumnMeta(tableName, columnName) {
    const tableMeta = state.columnMeta[tableName] || [];
    return tableMeta.find(c => c.name === columnName);
}

// Normalize data type for display
export function normalizeType(rawType) {
    const upper = (rawType || '').toUpperCase();
    if (upper.includes('VARCHAR') || upper.includes('TEXT') || upper.includes('CHAR')) return 'varchar';
    if (upper.includes('BIGINT')) return 'bigint';
    if (upper.includes('INT')) return 'integer';
    if (upper.includes('FLOAT') || upper.includes('DOUBLE') || upper.includes('DECIMAL')) return 'float';
    if (upper.includes('TIMESTAMP')) return 'timestamp';
    if (upper.includes('DATE') && !upper.includes('TIMESTAMP')) return 'date';
    if (upper.includes('BOOL')) return 'boolean';
    if (upper.includes('JSON')) return 'json';
    return 'varchar';
}
