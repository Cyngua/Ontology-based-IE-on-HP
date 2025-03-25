-- Table: Notes
CREATE TABLE Notes (
    note_id TEXT PRIMARY KEY,
    text TEXT
);

-- Table: Annotations
CREATE TABLE Annotations (
    note_id TEXT,
    start INTEGER,
    end INTEGER,
    concept_id INTEGER,
    PRIMARY KEY (note_id, start, end),
    FOREIGN KEY (note_id) REFERENCES Notes(note_id),
    FOREIGN KEY (concept_id) REFERENCES Concepts(concept_id)
);

-- Table: Concepts
CREATE TABLE Concepts (
    concept_id INTEGER,
    cui TEXT,
    name TEXT,
    semantic_types TEXT,
    atom_url TEXT,
    relations_url TEXT,
    PRIMARY KEY (concept_id, cui),
    FOREIGN KEY (semantic_types) REFERENCES Semantic_Groups(semantic_type)
);

-- Table: Semantic_Groups
CREATE TABLE Semantic_Groups (
    semantic_type TEXT,
    tui TEXT,
    group_name TEXT,
    group_abbr TEXT,
    PRIMARY KEY (semantic_type, tui)
);
