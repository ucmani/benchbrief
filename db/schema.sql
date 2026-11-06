create table dockets (
    id bigint primary key,
    case_name text,
    court_id text,
    assigned_to_str text,
    nature_of_suit text,
    date_filed date,
    created_at timestamptz default now()
);

create table clusters (
    id bigint primary key,
    docket_id bigint not null references dockets(id),
    case_name text,
    nature_of_suit text,
    judges text,
    date_filed date,
    created_at timestamptz default now()
);

create table opinions (
    id bigint primary key,
    cluster_id bigint not null references clusters(id),
    text_html text,
    text_plain text,
    extracted_by_ocr boolean,
    created_at timestamptz default now()
);

create table chunks (
    id uuid primary key default gen_random_uuid(),
    opinion_id bigint not null references opinions(id),
    strategy_name text not null,
    chunk_index int not null,
    section_label text,
    text text not null,
    metadata jsonb,
    created_at timestamptz default now()
);