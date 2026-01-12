/**
 * DuckDB WASM Integration
 * Handles database initialization, loading, and queries
 */

import { state, notify, normalizeType, isFactTable, isDimTable } from './state.js';

// DuckDB module reference
let duckdb = null;

/**
 * Initialize DuckDB WASM
 */
export async function initDuckDB() {
    // Dynamic import of DuckDB WASM
    duckdb = await import('https://cdn.jsdelivr.net/npm/@duckdb/duckdb-wasm@1.29.0/+esm');

    const JSDELIVR_BUNDLES = duckdb.getJsDelivrBundles();
    const bundle = await duckdb.selectBundle(JSDELIVR_BUNDLES);

    const workerUrl = URL.createObjectURL(
        new Blob([`importScripts("${bundle.mainWorker}");`], { type: 'text/javascript' })
    );

    const logger = new duckdb.ConsoleLogger();
    const worker = new Worker(workerUrl);
    state.db = new duckdb.AsyncDuckDB(logger, worker);
    await state.db.instantiate(bundle.mainModule, bundle.pthreadWorker);
    URL.revokeObjectURL(workerUrl);

    console.log('DuckDB WASM initialized');
}

/**
 * Load a database file
 */
export async function loadDatabase(file) {
    const buffer = await file.arrayBuffer();
    await state.db.registerFileBuffer(file.name, new Uint8Array(buffer));
    state.conn = await state.db.connect();
    await state.conn.query(`ATTACH '${escapeSQL(file.name)}' AS loaded_db`);
    state.dbPrefix = 'loaded_db.';

    // Load semantic model
    await loadSemanticModel();

    // Discover tables
    await discoverTables();

    // Build relationships from semantic model
    buildRelationships();

    notify('database-loaded');
}

/**
 * Load semantic model from database
 */
async function loadSemanticModel() {
    try {
        const result = await state.conn.query(
            `SELECT * FROM ${state.dbPrefix}meta_semantic_model ORDER BY sort_order`
        );
        state.semanticModel = result.toArray().map(row => ({
            tableName: row.table_name,
            tableType: row.table_type,
            tableDisplayName: row.table_display_name,
            columnName: row.column_name,
            columnType: row.column_type,
            dataType: row.data_type,
            displayName: row.display_name,
            defaultAggregation: row.default_aggregation,
            relatedTable: row.related_table,
            relatedColumn: row.related_column,
            isVisible: row.is_visible,
            isFilterable: row.is_filterable
        }));
        console.log(`Loaded semantic model with ${state.semanticModel.length} entries`);
    } catch (e) {
        console.warn('No semantic model found, using basic schema discovery');
        state.semanticModel = [];
    }
}

/**
 * Discover tables in the database
 */
async function discoverTables() {
    // Use information_schema which works in DuckDB WASM
    const result = await state.conn.query(`
        SELECT table_name
        FROM information_schema.tables
        WHERE table_catalog = 'loaded_db'
        ORDER BY table_name
    `);
    state.tables = result.toArray().map(r => r.table_name);
    state.factTables = state.tables.filter(isFactTable);
    state.dimTables = state.tables.filter(isDimTable);

    // Load column metadata for all tables
    for (const table of state.tables) {
        await loadTableColumns(table);
    }
}

/**
 * Load column metadata for a table
 */
async function loadTableColumns(tableName) {
    const result = await state.conn.query(`DESCRIBE ${state.dbPrefix}"${escapeIdentifier(tableName)}"`);

    state.columnMeta[tableName] = result.toArray().map(row => {
        const colName = row.column_name;
        const rawType = row.column_type || row.data_type;
        const meta = state.semanticModel.find(
            m => m.tableName === tableName && m.columnName === colName
        );

        return {
            name: colName,
            rawType: rawType,
            dataType: meta?.dataType || normalizeType(rawType),
            displayName: meta?.displayName || formatColumnName(colName),
            columnType: meta?.columnType || 'attribute',
            defaultAggregation: meta?.defaultAggregation,
            isVisible: meta?.isVisible !== false,
            isFilterable: meta?.isFilterable !== false,
            relatedTable: meta?.relatedTable,
            relatedColumn: meta?.relatedColumn
        };
    });
}

/**
 * Build relationship map from semantic model
 */
function buildRelationships() {
    state.relationships = {};

    // Group by fact table
    for (const meta of state.semanticModel) {
        if (!meta.relatedTable || !meta.relatedColumn) continue;
        if (!isFactTable(meta.tableName)) continue;

        if (!state.relationships[meta.tableName]) {
            state.relationships[meta.tableName] = [];
        }

        // Check if this relationship already exists
        const exists = state.relationships[meta.tableName].some(
            r => r.dimTable === meta.relatedTable && r.factCol === meta.columnName
        );

        if (!exists) {
            state.relationships[meta.tableName].push({
                dimTable: meta.relatedTable,
                factCol: meta.columnName,
                dimCol: meta.relatedColumn
            });
        }
    }

    console.log('Built relationships:', state.relationships);
}

/**
 * Execute a query and return results
 */
export async function executeQuery(sql) {
    state.lastQuery = sql;
    const result = await state.conn.query(sql);
    return result.toArray();
}

/**
 * Get distinct values for filtering
 */
export async function getDistinctValues(tableName, columnName, searchTerm = '') {
    const qualifiedTable = state.dbPrefix + tableName;
    let sql = `SELECT DISTINCT ${columnName} AS val FROM ${qualifiedTable} WHERE ${columnName} IS NOT NULL`;

    if (searchTerm) {
        sql += ` AND CAST(${columnName} AS VARCHAR) ILIKE '%${escapeSQL(searchTerm)}%'`;
    }
    sql += ` ORDER BY ${columnName} LIMIT 100`;

    const result = await state.conn.query(sql);
    return result.toArray().map(r => r.val);
}

// Helpers
export function escapeSQL(str) {
    return String(str).replace(/'/g, "''");
}

export function escapeIdentifier(name) {
    return String(name).replace(/"/g, '""');
}

function formatColumnName(name) {
    return name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}
