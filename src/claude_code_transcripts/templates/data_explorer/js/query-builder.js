/**
 * Query Builder with Semantic Joins
 * Builds SQL queries with automatic joins based on the semantic model
 */

import { state, getRelationships, notify } from './state.js';
import { escapeSQL, executeQuery } from './duckdb.js';

/**
 * Find a selected column by its alias
 */
function findColumnByAlias(alias) {
    return state.selectedColumns.find(c => (c.alias || c.column) === alias);
}

/**
 * Build and execute the current query
 */
export async function runQuery() {
    if (!state.baseTable || state.selectedColumns.length === 0) {
        state.data = [];
        state.totalRows = 0;
        state.lastQuery = '';
        notify('query-complete');
        return;
    }

    const sql = buildSelectQuery();
    state.lastQuery = sql;

    try {
        state.data = await executeQuery(sql);

        // Get total count
        const countSql = buildCountQuery();
        const countResult = await executeQuery(countSql);
        state.totalRows = Number(countResult[0]?.cnt || 0);

        notify('query-complete');
    } catch (e) {
        console.error('Query error:', e);
        state.data = [];
        state.totalRows = 0;
        notify('query-error');
    }
}

/**
 * Build the SELECT query with joins
 */
function buildSelectQuery() {
    const { selectClause, fromClause, whereClause, orderClause } = buildQueryParts();
    const offset = state.page * state.pageSize;

    return `${selectClause}\n${fromClause}\n${whereClause}${orderClause}\nLIMIT ${state.pageSize} OFFSET ${offset}`;
}

/**
 * Build COUNT query for pagination
 */
function buildCountQuery() {
    const { fromClause, whereClause } = buildQueryParts();
    return `SELECT COUNT(*) AS cnt\n${fromClause}\n${whereClause}`;
}

/**
 * Build query components (reusable for SELECT and COUNT)
 */
function buildQueryParts() {
    const baseAlias = 'f';
    const joins = [];
    const dimAliases = {}; // { dimTable: alias }
    let aliasCounter = 0;

    // Determine which dimension tables need to be joined
    const neededDims = new Set();
    for (const col of state.selectedColumns) {
        if (col.table !== state.baseTable) {
            neededDims.add(col.table);
        }
    }

    // Also check filters for dimension columns
    for (const [colAlias, values] of Object.entries(state.filters)) {
        if (values && values.length > 0) {
            const col = findColumnByAlias(colAlias);
            if (col && col.table !== state.baseTable) {
                neededDims.add(col.table);
            }
        }
    }

    // Build JOINs based on relationships
    const relationships = getRelationships(state.baseTable);
    for (const dimTable of neededDims) {
        const rel = relationships.find(r => r.dimTable === dimTable);
        if (rel) {
            const dimAlias = `d${++aliasCounter}`;
            dimAliases[dimTable] = dimAlias;
            joins.push(
                `LEFT JOIN ${state.dbPrefix}${dimTable} ${dimAlias} ON ${baseAlias}.${rel.factCol} = ${dimAlias}.${rel.dimCol}`
            );
        }
    }

    // Rebuild SELECT with correct aliases (now that we have dim aliases)
    const finalSelectParts = [];
    for (const col of state.selectedColumns) {
        let alias;
        if (col.table === state.baseTable) {
            alias = baseAlias;
        } else {
            alias = dimAliases[col.table];
        }
        if (alias) {
            const colAlias = col.alias || col.column;
            finalSelectParts.push(`${alias}.${col.column} AS "${colAlias}"`);
        }
    }

    // Build FROM clause
    let fromClause = `FROM ${state.dbPrefix}${state.baseTable} ${baseAlias}`;
    if (joins.length > 0) {
        fromClause += '\n' + joins.join('\n');
    }

    // Build WHERE clause
    const conditions = [];
    for (const [colAlias, values] of Object.entries(state.filters)) {
        if (values && values.length > 0) {
            const col = findColumnByAlias(colAlias);
            if (col) {
                let tableAlias = col.table === state.baseTable ? baseAlias : dimAliases[col.table];
                if (tableAlias) {
                    const escaped = values.map(v => `'${escapeSQL(String(v))}'`).join(',');
                    conditions.push(`${tableAlias}.${col.column} IN (${escaped})`);
                }
            }
        }
    }
    const whereClause = conditions.length > 0 ? 'WHERE ' + conditions.join(' AND ') : '';

    // Build ORDER clause
    let orderClause = '';
    if (state.sorts.length > 0) {
        const orderParts = state.sorts.map(s => {
            const col = findColumnByAlias(s.column);
            if (col) {
                const tableAlias = col.table === state.baseTable ? baseAlias : dimAliases[col.table];
                if (tableAlias) {
                    return `${tableAlias}.${col.column} ${s.dir.toUpperCase()}`;
                }
            }
            return `"${s.column}" ${s.dir.toUpperCase()}`;
        });
        orderClause = '\nORDER BY ' + orderParts.join(', ');
    }

    return {
        selectClause: 'SELECT ' + (finalSelectParts.length > 0 ? finalSelectParts.join(',\n       ') : '*'),
        fromClause,
        whereClause,
        orderClause
    };
}

/**
 * Add a column to the selection
 */
export function addColumn(tableName, columnName, alias = null) {
    // Check if already selected
    const existing = state.selectedColumns.find(
        c => c.table === tableName && c.column === columnName
    );
    if (existing) return;

    // Generate unique alias if needed
    const colAlias = alias || generateUniqueAlias(columnName);

    state.selectedColumns.push({
        table: tableName,
        column: columnName,
        alias: colAlias
    });

    // Track joined tables
    if (tableName !== state.baseTable) {
        state.joinedTables.add(tableName);
    }

    notify('columns-changed');
}

/**
 * Remove a column from the selection
 */
export function removeColumn(tableName, columnName) {
    const idx = state.selectedColumns.findIndex(
        c => c.table === tableName && c.column === columnName
    );
    if (idx >= 0) {
        state.selectedColumns.splice(idx, 1);

        // Check if we still need this table joined
        if (tableName !== state.baseTable) {
            const stillNeeded = state.selectedColumns.some(c => c.table === tableName);
            if (!stillNeeded) {
                state.joinedTables.delete(tableName);
            }
        }

        notify('columns-changed');
    }
}

/**
 * Check if a column is selected
 */
export function isColumnSelected(tableName, columnName) {
    return state.selectedColumns.some(
        c => c.table === tableName && c.column === columnName
    );
}

/**
 * Set the base fact table
 */
export function setBaseTable(tableName) {
    if (state.baseTable === tableName) return;

    state.baseTable = tableName;
    state.selectedColumns = [];
    state.joinedTables = new Set();
    state.filters = {};
    state.sorts = [];
    state.page = 0;

    notify('base-table-changed');
}

/**
 * Toggle sort on a column
 */
export function toggleSort(columnAlias) {
    const existing = state.sorts.find(s => s.column === columnAlias);
    if (existing) {
        if (existing.dir === 'asc') {
            existing.dir = 'desc';
        } else {
            state.sorts = state.sorts.filter(s => s.column !== columnAlias);
        }
    } else {
        state.sorts = [{ column: columnAlias, dir: 'asc' }];
    }
    state.page = 0;
    notify('sort-changed');
}

/**
 * Set filter on a column
 */
export function setFilter(columnAlias, values) {
    if (values && values.length > 0) {
        state.filters[columnAlias] = values;
    } else {
        delete state.filters[columnAlias];
    }
    state.page = 0;
    notify('filter-changed');
}

/**
 * Clear all filters
 */
export function clearFilters() {
    state.filters = {};
    state.page = 0;
    notify('filter-changed');
}

/**
 * Navigate pagination
 */
export function setPage(page) {
    const maxPage = Math.ceil(state.totalRows / state.pageSize) - 1;
    state.page = Math.max(0, Math.min(page, maxPage));
    notify('page-changed');
}

/**
 * Generate a unique column alias
 */
function generateUniqueAlias(baseName) {
    let alias = baseName;
    let counter = 1;
    while (state.selectedColumns.some(c => c.alias === alias)) {
        alias = `${baseName}_${counter++}`;
    }
    return alias;
}

/**
 * Format SQL for display with syntax highlighting
 * Returns array of { text, class } objects for safe DOM rendering
 */
export function formatSQLParts(sql) {
    if (!sql) return [{ text: '-- No query', class: 'text-gray-500' }];

    const parts = [];
    const keywords = ['SELECT', 'FROM', 'LEFT JOIN', 'JOIN', 'ON', 'WHERE', 'AND', 'OR', 'ORDER BY', 'LIMIT', 'OFFSET', 'AS', 'IN', 'GROUP BY', 'HAVING', 'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'DISTINCT'];
    const keywordSet = new Set(keywords.map(k => k.toUpperCase()));

    // Tokenize the SQL
    const tokens = tokenizeSQL(sql);

    for (const token of tokens) {
        if (keywordSet.has(token.toUpperCase())) {
            parts.push({ text: token, class: 'sql-keyword' });
        } else if (/^'.*'$/.test(token)) {
            parts.push({ text: token, class: 'sql-string' });
        } else if (/^\d+$/.test(token)) {
            parts.push({ text: token, class: 'sql-number' });
        } else if (/^[fd]\d*\.$/.test(token)) {
            parts.push({ text: token.slice(0, -1), class: 'sql-alias' });
            parts.push({ text: '.', class: '' });
        } else if (/^".*"$/.test(token)) {
            parts.push({ text: token, class: 'sql-table' });
        } else {
            parts.push({ text: token, class: '' });
        }
    }

    return parts;
}

/**
 * Simple SQL tokenizer
 */
function tokenizeSQL(sql) {
    const tokens = [];
    let current = '';
    let inString = false;
    let inQuote = false;

    for (let i = 0; i < sql.length; i++) {
        const char = sql[i];

        if (inString) {
            current += char;
            if (char === "'" && sql[i + 1] !== "'") {
                tokens.push(current);
                current = '';
                inString = false;
            }
        } else if (inQuote) {
            current += char;
            if (char === '"') {
                tokens.push(current);
                current = '';
                inQuote = false;
            }
        } else if (char === "'") {
            if (current) tokens.push(current);
            current = char;
            inString = true;
        } else if (char === '"') {
            if (current) tokens.push(current);
            current = char;
            inQuote = true;
        } else if (/\s/.test(char)) {
            if (current) tokens.push(current);
            tokens.push(char);
            current = '';
        } else if (/[(),]/.test(char)) {
            if (current) tokens.push(current);
            tokens.push(char);
            current = '';
        } else {
            current += char;
        }
    }

    if (current) tokens.push(current);
    return tokens;
}
