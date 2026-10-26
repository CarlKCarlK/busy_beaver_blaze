// Rows are states A, B, ...; columns are input symbols 0, 1, ... .
export function parseProgram(text) {
    const lines = text.trim().split(/\r?\n/).map(line => line.trim()).filter(Boolean);
    let rows;
    if (lines.length === 1) {
        rows = lines[0].split('_').map(section => {
            if (section.length % 3 !== 0) throw new Error('Each transition must have three characters.');
            return section.match(/.{3}/g) || [];
        });
    } else {
        const header = lines[0].split(/\s+/);
        const table = lines.slice(1).map(line => line.split(/\s+/));
        if (/^\d/.test(lines[0])) {
            if (!header.every((label, index) => label === String(index))) throw new Error('Expected symbol headings 0, 1, … .');
            rows = table.map((row, index) => {
                if (row[0] !== String.fromCharCode(65 + index) || row.length !== header.length + 1) throw new Error('Invalid state row.');
                return row.slice(1);
            });
        } else {
            if (!header.every((label, index) => label === String.fromCharCode(65 + index))) throw new Error('Expected state headings A, B, … .');
            table.forEach((row, index) => {
                if (row[0] !== String(index) || row.length !== header.length + 1) throw new Error('Invalid symbol row.');
            });
            rows = header.map((_, stateIndex) => table.map(row => row[stateIndex + 1]));
        }
    }
    const symbolCount = rows[0]?.length;
    if (!symbolCount || !rows.every(row => row.length === symbolCount)) throw new Error('Every state must have the same number of transitions.');
    if (!rows.flat().every(action => action === '---' || (/^[0-9][LR][A-Z]$/.test(action) && Number(action[0]) < symbolCount))) throw new Error('Invalid transition or write symbol.');
    return rows;
}

export function stateColoredProgram(text) {
    const rows = parseProgram(text);
    if (rows[0].length !== 2) throw new Error('This program requires Symbol coloring.');
    // The Rust parser uses single decimal digits and stores at most 100 actions.
    if (rows.length > 9) throw new Error('Writing state coloring currently supports at most nine states.');
    return rows.map((row, stateIndex) => Array.from({ length: rows.length + 1 }, (_, symbolIndex) => {
        const action = row[symbolIndex === 0 ? 0 : 1];
        if (action === '---') return action;
        return `${action[0] === '0' ? 0 : stateIndex + 1}${action.slice(1)}`;
    }).join('')).join('_');
}

export function statePalette(mode, stateCount) {
    const light = ['#ffffff', '#e69f00', '#0072b2', '#009e73', '#cc79a7', '#d55e00', '#56b4e9', '#8064a2', '#8c564b', '#bcbd22'];
    const dark = ['#ffffff', '#000000', '#555555', '#888888', '#aaaaaa', '#cccccc', '#333333', '#666666', '#999999', '#bbbbbb'];
    const light2 = ['#ffffff', '#d6b800', '#009fbb', '#e87500', '#c93baf', '#0072b2', '#009e73', '#8064a2', '#8c564b', '#666666'];
    const palette = mode === 'dark' ? dark : mode === 'light2' ? light2 : light;
    return new Uint8Array(palette.slice(0, stateCount + 1).flatMap(hex => [1, 3, 5].map(offset => parseInt(hex.slice(offset, offset + 2), 16))));
}
