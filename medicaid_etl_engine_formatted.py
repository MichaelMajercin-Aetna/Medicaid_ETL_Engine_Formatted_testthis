"""
FileName: medicaid_etl_engine.py
Description: Script used to transform and process raw data from plan audit and format into
             Inovalon ECDS layout. Script reads configuration file and extracts tables, queries,
             relevant columns for each source.
Args:
    --config: Path to config file to use.
    --mode: Which mode to use (data_refresh / previous_month / sample)
        data_refresh    -> pulls data from the last 4 years
        previous_month  -> pulls data from only the last month
        sample          -> pulls 10,000 records, useful for testing new sources
    --date_range: Used with data_refresh. Pulls data between two dates supplied as arguments.

Example usage:
    python medicaid_etl_engine.py --config ../configs/imm_configs/Config_Imm_IL_ICare.yaml --mode data_refresh
    python medicaid_etl_engine.py --config ../configs/imm_configs/Config_Imm_IL_ICare.yaml --mode data_refresh --date_range 2021-01-01 2022-01-01
    python medicaid_etl_engine.py --config ../configs/imm_configs/Config_Imm_IL_ICare.yaml --mode previous_month

Author: Michael Majercin
Date: 2025-03-28
Version: 2.0
"""

from datetime import date, datetime, timedelta
import argparse
import logging
import os
import time

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
import yaml

import measures_module
import quick_base_automation

# Create empty dict for storing summary data
summary_data = {}


def load_config(config_path):
    """
    Def:
        Function used to load config file.
    Args:
        config_path: Path to configuration file.
    Returns:
        Dictionary from parsed yaml file.
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def pull_medicaid_data(config, mode, date_range):
    """
    Def:
        Function used to build query based off configuration file, run query, and return planaudit
        data in dataframe. Mode is used to determine what date range.
    Args:
        config: Dictionary with configuration data.
        mode: Mode from config file.
        date_range: Custom date range.
    Returns:
        Dataframe containing raw medicaid data from planaudit.
    """
    # Read in config data
    server = config['connection']['server']
    db_name = config['connection']['database']
    raw_sql = config['sql_main']

    if mode == "previous_month":
        # start date = 1st day of previous month
        # EX: 2024-12-01
        start_date = date.today().replace(day=1) - relativedelta(months=1)
        # end date = last day of previous month
        # EX: 2024-12-31
        end_date = date.today().replace(day=1) - timedelta(days=1)
        # config top clause
        top_clause = ""
        # summary data
        summary_data['Query_Start_date'] = start_date
        summary_data['Query_End_date'] = end_date

    elif mode == "data_refresh":
        # 1st day of the year 4 years prior
        # should this be 2021-01-01?
        # EX: 2024-12-01
        start_date = date.today().replace(day=1) - relativedelta(months=1, years=4)
        # end date = last day of previous month
        end_date = date.today().replace(day=1) - timedelta(days=1)
        # config top clause
        top_clause = ""
        if date_range:
            start_date, end_date = date_range
            top_clause = ""
        # summary data
        summary_data['Query_Start_date'] = start_date
        summary_data['Query_End_date'] = end_date

    elif mode == "sample":
        # 1st day of the year 4 years prior (note: configured as 2 years in sample mode)
        # EX: 2024-12-01
        start_date = date.today().replace(day=1) - relativedelta(months=1, years=2)
        # end date = last day of previous month
        end_date = date.today().replace(day=1) - timedelta(days=1)
        # config top clause
        top_clause = "TOP 10000"

    logging.info(f"===== Mode Configured: {mode} =====")
    logging.info(
        f"===== Pulling source data between {start_date} and {end_date} ====="
    )

    conn = (
        'mssql+pyodbc://' + server + '/' + db_name + '?driver=SQL+Server+Native+Client+11.0'
    )

    try:
        # adding config here 8-6-2025
        main_df = run_query(raw_sql, conn, top_clause, start_date, end_date, config)
    except Exception:
        main_df = run_chunked_query(conn, raw_sql, top_clause, start_date, end_date)

    return main_df


def generate_year_chunks(overall_start, overall_end):
    """
    Def:
        Generator function used to supply date chunks to run_chunked_query function.
        Breaks up large date range unto 1 year chunks for larger sources.
    Args:
        overall_start: Start date from query date range.
        overall_end: End date from query date range.
    Yields:
        chunk_start, chunk_end: chunked date range increments.
    """
    chunk_start = overall_start
    while chunk_start < overall_end:
        chunk_end = min(chunk_start + relativedelta(years=1), overall_end)
        yield (chunk_start, chunk_end)
        chunk_start = chunk_end


def run_chunked_query(conn, query, top_clause, overall_start, overall_end):
    """
    Def:
        Function runs query with chunked dates in combination with generate_year_chunks function.
        For each date range supplied function pulls data from plan audit in Inovalon format, downcasting
        all datatypes to string, coalescing MemberKey in batches of 300_000 records at a time.
    Args:
        conn: Connection string for planaudit table.
        query: Main planaudit query.
        top_clause: Clause for select statement.
        overall_start: Start date.
        overall_end: End date.
    Returns:
        Dataframe with planaudit data in Inovalon format.
    """
    all_chunks = []

    inovalon_format = [
        'MemberKey', 'ProviderKey', 'ReferenceID', 'DOS', 'DOSThru', 'ProviderType',
        'ProviderTaxonomy', 'CPTPx', 'HCPCSPx', 'LOINC', 'SNOMED', 'ICDDX', 'ICDDX10',
        'RxNorm', 'CVX', 'Modifier', 'RxProviderFlag', 'PCPFlag', 'QuantityDispensed',
        'ICDPx', 'ICDPx10', 'SuppSource', 'Result', 'Ecdsaltid_00', 'CPTmod', 'HCPCSmod',
        'LOINCAnswer',
    ]

    for start_date, end_date in generate_year_chunks(overall_start, overall_end):
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')

        # testing for adhoc, remove try, except if causes issue and just use sql_query replace block.
        try:
            sql_query = (
                query
                .replace('{top_clause}', top_clause)
                .replace('{start_date}', str(start_date))
                .replace('{end_date}', str(end_date))
            )
        except Exception:
            print(
                'Running custom query, query is missing start date, end date, and or top clause'
            )

        print(f"Pulling data from {start_date} to {end_date}")

        for chunk in pd.read_sql(
            sql_query,
            conn,
            coerce_float=False,
            parse_dates=['DOS'],
            chunksize=300_000,
        ):
            # Coalesce MemberKey from alternate joins when present
            if "MemberKey1" in chunk.columns and "MemberKey2" in chunk.columns:
                chunk['MemberKey'] = chunk['MemberKey1'].fillna(chunk['MemberKey2'])
                chunk.drop(['MemberKey1', 'MemberKey2'], axis=1, inplace=True)

            # Reindex to ensure the chunk has ALL inovalon format columns
            all_needed_columns = chunk.columns.union(inovalon_format)
            chunk = chunk.reindex(columns=all_needed_columns, fill_value='')

            # List of all columns except DOS for casting
            columns_to_cast = [col for col in chunk.columns if col != 'DOS']
            # Downcast each column to string except for DOS
            chunk[columns_to_cast] = chunk[columns_to_cast].astype('string')

            # Fill any missing values with empty string to simplify downstream logic
            chunk = chunk.fillna('')

            all_chunks.append(chunk)
            # Debugging/visibility prints
            print('Processed chunk...')
            print(f'Rec count: {len(all_chunks) * 300_000}')
            print(chunk.dtypes)

    return pd.concat(all_chunks, ignore_index=True)


def run_query(query, conn, top_clause, start_date, end_date, config):
    """
    Def:
        Function runs query to pull data from planaudit. Query is executed in chunks of 300_000
        records at a time. All columns are downcasted to strings to save memory and simplify later
        merging. MemberKeys are coalesced into one column. Data is formatted into Inovalon ECDS layout.
    Args:
        conn: Connection string for planaudit table.
        query: Main planaudit query.
        top_clause: Clause for select statement.
        start_date: Start date.
        end_date: End date.
    Returns:
        Dataframe with planaudit data in Inovalon format.
    """
    all_chunks = []

    # testing for adhoc, remove try, except if causes issue and just use sql_query replace block.
    try:
        sql_query = (
            query
            .replace('{top_clause}', top_clause)
            .replace('{start_date}', str(start_date))
            .replace('{end_date}', str(end_date))
        )
    except Exception:
        print('Running custom query, query is missing start date, end date, and or top clause')

    # Config File Params
    summary_data['Main_SQL_Query'] = sql_query

    # try icd split logic
    try:
        icd_split = config['custom_logic']['icd_split']
        icd_col_name = config['custom_logic']['icd_params']['icd_col_name']
        icd_range = config['custom_logic']['icd_params']['icd_range']
        code_cols = config['filter_rules']['code_columns']
        meta_cols = [
            'MemberKey', 'ProviderKey', 'ReferenceID', 'DOS', 'DOSThru', 'ProviderType',
            'ProviderTaxonomy', 'CPTPx', 'HCPCSPx', 'LOINC', 'SNOMED', 'ICDDX', 'RxNorm', 'CVX',
            'Modifier', 'RxProviderFlag', 'PCPFlag', 'QuantityDispensed', 'ICDPx', 'ICDPx10',
            'SuppSource', 'Result', 'Ecdsaltid_00', 'CPTmod', 'HCPCSmod', 'LOINCAnswer',
            'MedicaidID',
        ]
    except Exception:
        icd_split = False

    # inovalon format
    inovalon_format = [
        'MemberKey', 'ProviderKey', 'ReferenceID', 'DOS', 'DOSThru', 'ProviderType',
        'ProviderTaxonomy', 'CPTPx', 'HCPCSPx', 'LOINC', 'SNOMED', 'ICDDX', 'ICDDX10', 'RxNorm',
        'CVX', 'Modifier', 'RxProviderFlag', 'PCPFlag', 'QuantityDispensed', 'ICDPx', 'ICDPx10',
        'SuppSource', 'Result', 'Ecdsaltid_00', 'CPTmod', 'HCPCSmod', 'LOINCAnswer',
    ]

    print(sql_query)

    for chunk in pd.read_sql(
        sql_query,
        conn,
        coerce_float=False,
        parse_dates=['DOS'],
        chunksize=300_000,
    ):
        # coalescing multiple member joins into final 'MemberKey' column
        if "MemberKey1" in chunk.columns and "MemberKey2" in chunk.columns:
            chunk['MemberKey'] = chunk['MemberKey1'].fillna(chunk['MemberKey2'])
            chunk.drop(['MemberKey1', 'MemberKey2'], axis=1, inplace=True)

        # ICD Split Logic
        if icd_split:
            print('Inside ICD SPLIT Function')
            chunk = normalize_icd_codes(
                chunk, code_cols, icd_col_name, icd_range, meta_cols
            )

        # Reindex to ensure the chunk has ALL inovalon format columns, blank columns will contain
        # empty string
        all_needed_columns = chunk.columns.union(inovalon_format)
        chunk = chunk.reindex(columns=all_needed_columns, fill_value='')

        # List of all columns except DOS for casting
        columns_to_cast = [col for col in chunk.columns if col != 'DOS']
        # Dynamically down cast each column to string except for DOS
        chunk[columns_to_cast] = chunk[columns_to_cast].astype('string')

        # Fill any missing values with empty string to simplify down stream logic
        chunk = chunk.fillna('')

        all_chunks.append(chunk)
        # printing for debugging/visibility, can be removed
        print('Processed chunk...')
        print(f'Rec count: {len(all_chunks) * 300_000}')
        print(chunk.dtypes)

    return pd.concat(all_chunks, ignore_index=True)


def dedupe_from_ytd(config, mode, current_month_df):
    """
    Def:
        Pull all medicaid data for source year to date (Jan 1st - last day of 2 months prior to current
        date). Remove duplicates from current_month_df using year to date data. Returns unique data and
        duplicate data in two seperate dataframes.
    Args:
        server: Server used in connection string (MBUPRODALT).
        db_name: Database used in connection string (HEDIS_STAGE).
        current_month_df: Removes duplicates from this df.
    Returns:
        final_unique: DF with all duplicate records removed.
        final_duplicate: DF containing removed duplicate records (same schema as current_month_df).
    """
    print('Removing Duplicates...')

    server = config['connection']['server']
    db_name = config['connection']['database']
    raw_sql = config['sql_dedupe']
    dedupe_columns = config['filter_rules']['dedupe_columns']
    summary_data['DeDupe_Columns'] = dedupe_columns

    # Preserve original column order/schema for clean outputs
    left_cols = list(current_month_df.columns)

    if mode == "previous_month":
        top_clause = ""
    elif mode == "data_refresh":
        top_clause = ""
    elif mode == "sample":
        top_clause = "TOP 1000"

    if mode == "previous_month" or mode == "sample":
        # Deduping against Year-To-Date data only (per 02/06/2025 guidance).
        # Start: Jan 1 of current year (dynamic, not hard-coded).
        # End: last day of two months prior (as in original logic).
        start_date = date(date.today().year, 1, 1)
        end_date = (date.today().replace(day=1) - timedelta(days=1)) - relativedelta(months=1)

        # Updating SQL Query
        sql_query = (
            raw_sql
            .replace('{top_clause}', top_clause)
            .replace('{start_date}', str(start_date))
            .replace('{end_date}', str(end_date))
        )

        logging.info(
            f"===== Checking for duplicates from data between {start_date} and {end_date} ====="
        )

        conn = (
            'mssql+pyodbc://' + server + '/' + db_name + '?driver=SQL+Server+Native+Client+11.0'
        )
        print(sql_query)
        query = sql_query

        duplicate_data = []
        current_df = current_month_df.copy()

        # Removing duplicates in chunks to work within memory constraints
        for chunk in pd.read_sql(
            query, conn, coalesce_float=False if 'coalesce_float' in pd.read_sql.__code__.co_varnames else False,
            coerce_float=False, parse_dates=['DOS'], chunksize=500_000
        ):
            print('Processing Chunk....')

            # NEW TESTING 08-06-2025
            # try icd split logic
            try:
                icd_split = config['custom_logic']['icd_split']
                icd_col_name = config['custom_logic']['icd_params']['icd_col_name']
                icd_range = config['custom_logic']['icd_params']['icd_range']
                code_cols = config['filter_rules']['code_columns']
                meta_cols = ['DOS', 'LOINC', 'Result', 'SNOMED', 'MedicaidID', 'CPTPx', 'HCPCSPx']
            except Exception:
                icd_split = False

            # ICD Split Logic
            if icd_split:
                print('Inside ICD SPLIT Function')
                chunk = normalize_icd_codes(
                    chunk, code_cols, icd_col_name, icd_range, meta_cols
                )

            # List of all columns except DOS for casting
            columns_to_cast = [col for col in chunk.columns if col != 'DOS']

            # Down cast each column to string except for DOS
            chunk[columns_to_cast] = chunk[columns_to_cast].astype('string')

            # 1 Left merge with indicator for slicing
            merged_df = pd.merge(
                current_df,
                chunk,
                how='left',
                on=dedupe_columns,
                indicator=True,
            )

            # 2 Slice duplicates (RETURN ONLY LEFT/ORIGINAL COLUMNS)
            dup_left = merged_df.loc[merged_df['_merge'] == 'both', left_cols].copy()

            # 3 Slice unique records (KEEP ONLY LEFT/ORIGINAL COLUMNS)
            still_unique = merged_df.loc[merged_df['_merge'] == 'left_only', left_cols].copy()

            # 4 Append duplicates
            if not dup_left.empty:
                duplicate_data.append(dup_left)

            # 5 For next iteration, only keep unique records (with clean schema)
            current_df = still_unique

        final_unique = current_df

        # Combine all duplicates safely (handle empty case)
        if duplicate_data:
            final_duplicate = pd.concat(duplicate_data, ignore_index=True)
            print(f"Real Duplicate Count: {len(final_duplicate)}")
            final_duplicate = final_duplicate.drop_duplicates()
        else:
            # Empty DF with same schema as input
            final_duplicate = current_month_df.iloc[0:0].copy()

        # Tag fallout reason (add the column if it doesn't exist yet)
        final_duplicate['Fallout Reason'] = 'Duplicate Record'

        return final_unique, final_duplicate

    if mode == "data_refresh":
        # data_refresh dedupe: drop duplicates within the dataset itself
        subset_columns = dedupe_columns
        current_df = current_month_df.copy()

        duplicates = current_df[current_df.duplicated(subset=subset_columns, keep='first')][left_cols].copy()
        duplicates['Fallout Reason'] = 'Duplicate Record'
        print(f"Real Duplicate Count: {len(duplicates)}")

        main_df = current_df.drop_duplicates(subset=subset_columns, keep='first')
        # Ensure main_df retains original column order/schema
        main_df = main_df[left_cols]

        return main_df, duplicates


def get_value_set(server, db_name, table_name):
    """
    Def:
        Function to query Medicaid valueset.
    Args:
        server: sql server.
        db_name: database name.
        table_name: table name.
    Returns:
        Dataframe containing valueset.
    """
    conn = (
        'mssql+pyodbc://' + server + '/' + db_name + '?driver=SQL+Server+Native+Client+11.0'
    )

    query = f"""
                SELECT distinct 
                    Code, 
                    "Code System"
                FROM {table_name}
             """

    value_set = pd.read_sql(query, conn, coerce_float=False)
    # List of all columns except DOS for casting
    columns_to_cast = [col for col in value_set.columns if col != 'DOS']

    # Down cast each column to string except for DOS
    value_set[columns_to_cast] = value_set[columns_to_cast].astype('string')

    return value_set


def get_value_set_flatFile():
    """
    Def:
        Function to read new 2025 value set into dataframe.
    Args:
        none.
    Returns:
        Dataframe containing valueset.
    """
    return pd.read_excel('./value_set/Value_Set_Raw_25.xlsx', dtype='string')


def value_set_check(month_df, month_code_to_check, value_set, value_set_code_to_check):
    """
    Def:
        Function will find bad codes by checking if they exist in the value_set dataframe. Bad codes
        will be stored in bad_codes_df for reporting purposes. Bad codes will be replaced with a
        empty string in the month_df as to avoid outputting invalid codes.
    Args:
        month_df: month data frame.
        month_code_to_check: column name of code to check.
        value_set: value set data frame.
        value_set_code_to_check: column name of code to check.
    Returns:
        cleaned_df: dataframe with bad codes replaced with empty string.
        bad_codes_df: dataframe with rows where bad codes were found for tracking purposes.
    """
    # 1 Copy month to avoid mutating the original
    month_copy = month_df.copy()

    # 2 Filter valueset to only merge on relevant columns
    value_set_code = value_set.loc[value_set['Code System'].isin(value_set_code_to_check)]
    value_set_code = value_set_code.drop_duplicates(subset=['Code'])

    # 3 Left merge with indicator to detect matches vs mismatches
    merged_df = pd.merge(
        month_copy,
        value_set_code,
        how='left',
        left_on=month_code_to_check,
        right_on='Code',
        indicator='code_check_indicator',
    )

    # 4 Identify and extract bad codes (where code not found in value_set)
    bad_codes_df = merged_df.loc[merged_df['code_check_indicator'] == 'left_only'].copy()
    bad_codes_df[f'Bad {month_code_to_check}'] = 'Missing in ValueSet, Code Removed'

    # 5 Overwrite invalid codes with ''
    merged_df[month_code_to_check] = merged_df[month_code_to_check].mask(
        merged_df['code_check_indicator'] == 'left_only', ''
    )

    # 6 Drop extra columns
    merged_df.drop(columns=['Code', 'Code System', 'code_check_indicator'], inplace=True)

    cleaned_df = merged_df

    return cleaned_df, bad_codes_df


def show_memory_usage(df):
    """
    Def:
        Function checks memory usage in GB of a dataframe.
    Args:
        df: Dataframe to check memory usage.
    Returns:
        None.
    """
    # Memory usage of each col
    memory_per_column = df.memory_usage(deep=True)
    # total memory usage in bytes
    total_memory = memory_per_column.sum()
    # convert to GB
    print(f"memory used: {total_memory / (1024 ** 3):.2f} GB")


def pull_il_crosswalk(server, db_name):
    """
    Def:
        Function to pull IL crosswalk file used for mapping vaccine IDs to medical codes.
    Args:
        server: Server for connection string.
        db_name: Database for connection string.
    Returns:
        Dataframe containing crosswalk data.
    """
    conn = (
        'mssql+pyodbc://' + server + '/' + db_name + '?driver=SQL+Server+Native+Client+11.0'
    )

    query = '''
            SELECT
            VACCINE_ID,
            VACCINE_NAME,
            VACCINE_CVX_CODE as CVX,
            VACCINE_CPT_CODE as CPTPx
            FROM [PlanAudit_IL].[AD].[ImImmunRefICare]
            '''

    return pd.read_sql(query, conn, coerce_float=False)


def pull_fl_crosswalk():
    """
    Def:
        Function to read Florida crosswalk file.
    Args:
        None.
    Returns:
        Dataframe containing Florida crosswalk file.
    """
    return pd.read_csv(
        "../Imm_Transformations/FL_IMMResponse/0_Refs/FL_Imm_CVX_Vax_type_mapping.csv",
        dtype='string',
        sep='|',
    )


# NEW TESTING 03/19/2025

def signify_mapping(df):
    """
    Def:
        Function to read new 2025 value set into dataframe.
    Args:
        none.
    Returns:
        Dataframe containing valueset.
    """
    sig_mapping = pd.read_excel('./mappings/Medicaid_Signify_Mapping.xlsx', dtype='string')
    df.drop(columns='LOINCAnswer', inplace=True)

    # Formatting pre merge:
    df['LOINCAnswer_txt'] = df['LOINCAnswer_txt'].str.strip().str.upper()
    sig_mapping['LOINCAnswer_txt'] = sig_mapping['LOINCAnswer_txt'].str.strip().str.upper()
    # Extracting sub strings
    df['LOINCAnswer_txt'] = (
        df['LOINCAnswer_txt']
        .str.extract(r'(I DO NOT HAVE A STEADY PLACE TO LIVE)')[0]
        .fillna(df['LOINCAnswer_txt'])
    )

    merged = df.merge(sig_mapping, on=['LOINCAnswer_txt', 'LOINC'], how='left')
    # merged['LOINC'] = merged['LOINC'].fillna(merged['LOINC_2'])
    merged.to_csv('ADHOC_POST_SIG.csv', index=False)
    merged.drop(columns='LOINCAnswer_txt', inplace=True)

    return merged


def clean_results(df):
    """Apply LOINC-based cleaning to Result values for a defined allowlist."""
    wanted_loincs = [
        '17855-8', '17856-6', '4548-4', '4549-2', '96595-4', '75995-1', '8453-3', '8462-4', '8496-2',
        '8514-2', '8515-9', '75997-7', '8459-0', '8480-6', '8508-4', '8546-4', '8547-2', '101351-5',
        '71802-3', '88122-7', '88123-5', '88124-3', '89569-8', '92358-1', '93030-5', '93031-3',
        '93033-9', '93668-2', '93669-0', '93671-6', '95251-5', '95264-8', '95399-2', '95400-8',
        '96434-6', '96441-1', '96778-6', '98976-4', '98977-2', '98978-0', '99134-9', '99135-6',
        '99550-6', '99553-0', '99594-4', '44261-6', '89204-2', '44261-6', '48544-1', '48545-8',
        '55758-7', '71777-7', '71965-8', '89204-2', '89205-9', '89208-3', '89209-1', '90221-3',
        '90853-3', '99046-5',
    ]

    df.loc[~df['LOINC'].isin(wanted_loincs), 'Result'] = ''
    return df


def populate_reference_id(df, ref_file):
    """
    Populate sequential ReferenceID values using a persisted file-based counter.
    This function mutates and returns the same DataFrame with a 'ReferenceID' column.
    """
    if os.path.exists(ref_file):
        with open(ref_file, 'r') as f:
            last_id_str = f.read().strip()

            if not last_id_str:
                last_number = 0
            else:
                last_number = int(last_id_str[3:])
    else:
        last_number = 0

    start_val = last_number + 1
    end_val = start_val + len(df)
    new_ids = [f"28I{i:019d}" for i in range(start_val, end_val)]

    df['ReferenceID'] = new_ids

    new_max_id = new_ids[-1] if new_ids else f"28I{last_number:019d}"

    with open(ref_file, 'w') as f:
        f.write(new_max_id)

    return df


# Testing New code for MIHIN 08/06/2025

def normalize_icd_codes(df, code_cols, icd_col_name, icd_range, meta_cols):
    """Explode wide ICD columns to a single ICDDX10 column while preserving meta columns."""
    all_needed_columns = df.columns.union(meta_cols)
    df = df.reindex(columns=all_needed_columns, fill_value='')

    diag_cols = [f'{icd_col_name}{i}' for i in range(*icd_range)]

    # Melt wide ICD codes into single col
    df_melted = df.melt(
        id_vars=meta_cols,
        value_vars=diag_cols,
        # var_name='DiagColumn',
        value_name='ICDDX10',
    )

    # new testing to fix merge issue
    df_melted = df_melted.drop(columns='variable')
    # Drop empty or null ICCDx10 values (intentionally left commented out per original logic)
    # df_melted = df_melted[df_melted['ICDDX10'].notna() & (df_melted['ICDDX10'] != '')]

    # Identify the first ICCDx10 record per patient/DOS
    df_melted['row_number'] = df_melted.groupby(meta_cols).cumcount()

    # Clear extra code fields in rows after the first row
    df_melted.loc[df_melted['row_number'] > 0, code_cols] = ''

    return df_melted.drop(columns='row_number')

"""
New tst is icd_code split function
def normalize_icd_codes(df, code_cols, icd_col_name, icd_range, meta_cols):
    ""
    Explode wide ICD columns (e.g., diagi1..diagi37) into a single 'ICDDX10' column.

    Behavior:
      - Only output rows for non-empty ICD codes (no explosion from blanks).
      - Keep other codes (code_cols) only on the *first* ICD row per original record.
      - If a record has no ICDs at all, pass it through once with ICDD X10 = ''.
      - Preserve DOS dtype; cast other cols to string where feasible.

    Args:
        df           : Input DataFrame.
        code_cols    : Columns like CPT/HCPCS/LOINC/SNOMED to keep only on first ICD row.
        icd_col_name : Base name for ICD columns (e.g., 'diagi').
        icd_range    : Tuple like (1, 38) -> diagi1..diagi37.
        meta_cols    : Columns to always carry through (member IDs, DOS, etc.).

    Returns:
        DataFrame with columns: meta_cols + code_cols + ['ICDDX10'].
    ""
    # Work on a copy; ensure needed cols exist (fill missing with '')
    df = df.copy()
    needed = sorted(set(meta_cols) | set(code_cols))
    df = df.reindex(columns=df.columns.union(needed), fill_value='')

    # Build list of ICD columns that actually exist
    diag_cols_all = [f"{icd_col_name}{i}" for i in range(*icd_range)]
    diag_cols = [c for c in diag_cols_all if c in df.columns]
    if not diag_cols:
        # No ICD columns at all -> passthrough once with empty ICD
        out = df[needed].copy()
        out["ICDDX10"] = ''
        return out

    # Flag rows that have at least one non-empty ICD code
    has_icd = (df[diag_cols].astype(str).apply(lambda s: s.str.strip() != '')).any(axis=1)

    # Passthrough rows (no ICDs): single row with ICDD X10='' and original codes kept
    passthrough = df.loc[~has_icd, needed].copy()
    passthrough["ICDDX10"] = ''

    # Rows with ICDs: melt ONLY non-empty values
    temp_row_id = "__row_id__"
    df[temp_row_id] = df.index  # stable key for grouping/ranking

    # Melt wide -> long
    long = df.loc[has_icd, [temp_row_id] + needed + diag_cols].melt(
        id_vars=[temp_row_id] + needed,
        value_vars=diag_cols,
        value_name="ICDDX10"
    )

    # Keep only non-empty ICDs
    long["ICDDX10"] = long["ICDDX10"].astype(str)
    long = long[long["ICDDX10"].str.strip() != ""]
    # Drop the melt var column; we don't need which diag slot it was
    if "variable" in long.columns:
        long = long.drop(columns=["variable"])

    # Rank ICDs within each original row to identify the "first" ICD
    long["__icd_rank__"] = long.groupby(temp_row_id).cumcount()

    # Blank out other-code columns (code_cols) on additional ICDs (>0)
    if code_cols:
        mask_extra = long["__icd_rank__"] > 0
        for c in code_cols:
            long.loc[mask_extra, c] = ''

    # Final select, drop helpers
    out_cols = needed + ["ICDDX10"]
    long = long[out_cols].copy()

    # Combine ICD long rows with passthrough (no-ICD) rows
    result = pd.concat([long, passthrough], ignore_index=True)

    # Keep DOS dtype; cast other columns to string for consistency
    for c in out_cols:
        if c != "DOS":  # don't force DOS -> string
            try:
                result[c] = result[c].astype('string')
            except Exception:
                # If a column cannot be cast cleanly (e.g., all ''), leave as-is
                pass

    return result
"""

# Testing new code for MIHIN 08/06/2025

def uniquify_alt_ids(df):
    """Make Ecdsaltid_00 values unique by suffixing duplicates with a sequence number."""
    df = df.copy()
    df["Ecdsaltid_00"] = (
        df["Ecdsaltid_00"].astype(str)
        + df.groupby("Ecdsaltid_00").cumcount().replace(0, '').astype(str).radd('_').replace('_', '')
    )

    return df

#Updated ValueSet Check Code to be faster 08/12/2025
def build_value_set_index(value_set_df):
    """
    Build {code_system: set_of_valid_codes} from your value_set dataframe.
    Expects columns: ['Code', 'Code System'].
    """
    vs = value_set_df.copy()
    vs['Code'] = vs['Code'].astype(str).str.strip()
    vs['Code System'] = vs['Code System'].astype(str).str.strip()

    index = {}
    for sys_name, grp in vs.groupby('Code System'):
        index[sys_name] = set(grp['Code'].dropna().astype(str))
    return index


def run_value_set_checks(df, value_set_index):
    """
    Validate common code columns against the value set index.
    - Invalid codes are set to '' (same behavior as before).
    - Returns:
        cleaned_df
        invalid_codes_df : combined fallout rows with 'Fallout Reason'
        codes_removed_df : summary per code type of distinct removed codes + counts
    """
    import pandas as pd

    # Inner helper so only two top-level functions are added
    def _value_set_check_fast(df_in, code_col, systems, fallback_reason):
        # If the column doesn't exist, return empty fallout & summary
        if code_col not in df_in.columns:
            empty_fallout = pd.DataFrame(columns=list(df_in.columns) + ['Fallout Reason'])
            empty_summary = pd.DataFrame({'Code Type': [code_col],
                                          'Removed Codes': [[]],
                                          'Removed Count': [0]})
            return df_in, empty_fallout, empty_summary

        out = df_in.copy()
        s = out[code_col].astype(str)

        # Work only on non-empty entries
        non_empty_mask = s.str.strip() != ''
        if not non_empty_mask.any():
            empty_fallout = pd.DataFrame(columns=list(df_in.columns) + ['Fallout Reason'])
            empty_summary = pd.DataFrame({'Code Type': [code_col],
                                          'Removed Codes': [[]],
                                          'Removed Count': [0]})
            return out, empty_fallout, empty_summary

        # Allowed codes across requested systems (case-insensitive compare)
        allowed = set().union(*(value_set_index.get(sys, set()) for sys in systems))
        allowed_upper = {str(x).strip().upper() for x in allowed}
        s_cmp = s.str.strip().str.upper()

        invalid_mask = non_empty_mask & (~s_cmp.isin(allowed_upper))

        # Fallout rows: same schema + reason
        fallout = out.loc[invalid_mask].copy()
        if not fallout.empty:
            fallout['Fallout Reason'] = fallback_reason
        else:
            fallout = pd.DataFrame(columns=list(df_in.columns) + ['Fallout Reason'])

        # Summary of distinct removed codes for this type
        removed_codes = sorted(set(s[invalid_mask].str.strip()))
        summary = pd.DataFrame({
            'Code Type': [code_col],
            'Removed Codes': [removed_codes],
            'Removed Count': [len(removed_codes)],
        })

        # Blank invalids in output (preserves existing behavior)
        out.loc[invalid_mask, code_col] = ''

        return out, fallout, summary

    # Which columns map to which code systems
    code_system_map = {
        'CVX':     ['CVX'],
        'CPTPx':   ['CPT', 'CPT-CAT-II'],
        'ICDDX10': ['ICD10PCS', 'ICD10CM'],
        'LOINC':   ['LOINC'],
        'SNOMED':  ['SNOMED CT US Edition'],
    }

    all_fallouts = []
    all_summaries = []
    out_df = df

    for col, systems in code_system_map.items():
        out_df, fallout_df, summary_df = _value_set_check_fast(
            out_df,
            code_col=col,
            systems=systems,
            fallback_reason=f'Invalid {col} (not in ValueSet)',
        )
        if not fallout_df.empty:
            all_fallouts.append(fallout_df)
        all_summaries.append(summary_df)

    invalid_codes_df = (
        pd.concat(all_fallouts, ignore_index=True) if all_fallouts
        else pd.DataFrame(columns=list(df.columns) + ['Fallout Reason'])
    )
    codes_removed_df = pd.concat(all_summaries, ignore_index=True)

    # Make the removed codes human-readable without lambdas
    codes_removed_df['Removed Codes'] = codes_removed_df['Removed Codes'].map(', '.join)

    return out_df, invalid_codes_df, codes_removed_df



def run_etl(config, mode, date_range):
    """Main ETL orchestration: pull -> filter -> enrich -> validate -> shape -> export."""
    # Testing preprocessed file
    try:
        flatfile = config['custom_logic']['file_name']
    except Exception:
        flatfile = ''

    print(f'FlatFile Arg: {flatfile}')

    if flatfile == '':
        # 1. Pull source data from PlanAudit
        #############################
        main_df = pull_medicaid_data(config, mode, date_range)
        summary_data['Input_Record_Count'] = len(main_df)
        logging.info(f"===== Record count for load month: {len(main_df)} =====")
        print(f'Pre DeDupe Length: {len(main_df)}')
        print('cols after import:')
        print(main_df.columns)
        #############################

        # 1.2 Clean Results column (testing 06/04/2025)
        #############################
        main_df = clean_results(main_df)
        # 2. Remove duplicates
        #############################
        main_df, duplicates = dedupe_from_ytd(config, mode, main_df)
        summary_data['Duplicate_Count'] = len(duplicates)
        print('Cols after dedupe')
        print(main_df.columns)
        print(f'Main {len(main_df)}')
        print(f'Duplicates {len(duplicates)}')
        print(f'De Duplicates {len(duplicates.drop_duplicates())}')
        #############################
    else:
        main_df = pd.read_csv(flatfile, dtype='string')
        main_df['DOS'] = pd.to_datetime(main_df['DOS'], errors='coerce')
        main_df.fillna('', inplace=True)
        summary_data['Input_Record_Count'] = len(main_df)
        summary_data['Query_Start_date'] = 'FlatFile'
        summary_data['Query_End_date'] = 'FlatFile'
        summary_data['Main_SQL_Query'] = 'FlatFile'
        summary_data['DeDupe_Columns'] = config['filter_rules']['dedupe_columns']
        print(main_df.dtypes)

    # 3. Member match filtering
    #############################
    # Filter invalid members
    print(main_df.columns)
    print('Starting Member Filtering')
    non_member_match = main_df.loc[main_df['MemberKey'].str.strip() == ''].copy()
    non_member_match['Fallout Reason'] = 'Member Not Found'
    print(f'Member Exclusion Count: {len(non_member_match)}')

    # Remove missing code records in main df
    main_df = main_df[(main_df['MemberKey'].str.strip() != '')]
    logging.info("===== Member filtering complete =====")
    logging.info(f"===== Bad member count: {len(non_member_match)} =====")
    summary_data['Bad_Member_Count'] = len(non_member_match)
    print('Finished Member Matching')
    #############################

    # IL - ICARE Custom Logic
    #############################
    source_name = config['source_id']
    if source_name == "IL_Icare":
        # IL custom logic here
        # dropping CVX and CPTPx to avoid ambigous rename, these will be readded by merge
        main_df.drop(columns=['CPTPx', 'CVX'], inplace=True)
        print(f"Main DF Length before Xwalk merge {len(main_df)}")
        # pull crosswalk
        crosswalk = pull_il_crosswalk('SRVQNXTRPTILPROD', 'PlanAudit_IL')

        # Merge matched member df with medical code crosswalk table
        main_df = main_df.merge(
            crosswalk,
            left_on=main_df['VACCINE_ID'].astype('string'),
            right_on=crosswalk['VACCINE_ID'].astype('string'),
            how='left',
        )
        print(f"Main DF Length AFTER Xwalk merge {len(main_df)}")
        print(main_df.columns)
        # Fill missing value with empty string for simpler filtering logic
        main_df = main_df.fillna('')
    ##############################

    # FL - StateResponse Custom Logic
    #############################
    if source_name == "FL_ImmResponse":
        # dropping CVX to avoid ambiguous rename, will be readded by merge
        main_df.drop(columns=['CVX'], inplace=True)

        # pull crosswalk
        crosswalk = pull_fl_crosswalk()

        # Merge matched member df with medical code crosswalk table
        main_df = main_df.merge(crosswalk, on='VaccinationType', how='left')
        print(main_df.columns)
        # Fill missing value with empty string for simpler filtering logic
        main_df = main_df.fillna('')
    ##############################

    # NEW TESTING 03/19/2025 — Signify Custom Logic
    ##############################
    # main_df = signify_mapping(main_df)

    try:
        print('trying custom logic')
        custom_logic = config['custom_logic']['function']
        print(f'Value: {custom_logic}')

        if custom_logic == 'Signify':
            main_df = signify_mapping(main_df)
    except Exception:
        print('No Custom Logic in config')

    try:
        print('Trying measure logic')
        measure_logic = config['custom_logic']['measures']
        print(f'Measures found: {measure_logic}')
        if 'GSD' in measure_logic:
            main_df, measure_fallout = measures_module.check_loinc_gsd(
                main_df, 'LOINC', 'Result'
            )
            print(
                f'Post measure check: main df: {len(main_df)}, measure_fallout count: {len(measure_fallout)}'
            )
        else:
            print('GSD is not in measure logic')
        if 'CBP/BPD' in measure_logic:
            main_df, measure_fallout = measures_module.check_loinc_cbp_bpd_dev(
                main_df, 'LOINC', 'Result'
            )
            print(
                f'Post measure check: main df: {len(main_df)}, measure_fallout count: {len(measure_fallout)}'
            )
        else:
            print('CBP/BPD is not in measure logic')
    except Exception:
        print('Meausre Logic Not Run...')

    ##############################

    # 4. Valid codes check
    #############################
    check_codes = config['filter_rules']['code_columns']
    summary_data['Code_Columns'] = check_codes

    # Pull valueset
    # value_set = get_value_set('MBUPRODALT', 'HEDIS_Reports', 'HCE.HEDIS_ValueSets_MY24')
    value_set = get_value_set_flatFile()

    # Formatting CVX before valueset check to match with valueset
    mask = main_df['CVX'] != ''
    main_df.loc[mask, 'CVX'] = main_df.loc[mask, 'CVX'].astype(str).str.zfill(2)

    # Formatting ICD Codes before valueset (Testing 4/17/2025)
    icd_mask = (
        (~main_df['ICDDX10'].str.contains(r'\.', regex=True)) & (main_df['ICDDX10'] != '')
    )
    main_df.loc[icd_mask, 'ICDDX10'] = (
        main_df.loc[icd_mask, 'ICDDX10'].str[:3]
        + '.'
        + main_df.loc[icd_mask, 'ICDDX10'].str[3:]
    )

    '''
    # Value set checks
    # *Need to revisit this logic; mapping for Codes to consolidate into 1 function?
    main_df, bad_cvx_codes_df = value_set_check(main_df, 'CVX', value_set, ['CVX'])
    main_df, bad_cpt_codes_df = value_set_check(
        main_df, 'CPTPx', value_set, ['CPT', 'CPT-CAT-II']
    )
    # New code untested
    main_df, bad_ICD10_codes_df = value_set_check(
        main_df, 'ICDDX10', value_set, ['ICD10PCS', 'ICD10CM']
    )
    main_df, bad_LOINC_codes_df = value_set_check(main_df, 'LOINC', value_set, ['LOINC'])
    main_df, bad_SNOMED_codes_df = value_set_check(
        main_df, 'SNOMED', value_set, ['SNOMED CT US Edition']
    )
    # Add HCPCs
    '''
    # Build index + run checks
    value_set_index = build_value_set_index(value_set)
    main_df, invalid_codes_df, codes_removed_df = run_value_set_checks(main_df, value_set_index)

    # Optional exports
    # invalid_codes_df: all invalid code rows together (use Fallout Reason to filter)
    # codes_removed_df: per code type list of distinct removed codes + counts

    logging.info(f"Invalid codes found: {len(invalid_codes_df)}")
    logging.info("\n" + str(codes_removed_df))

    logging.info("===== Value Set Check complete =====")

    '''
    # ! Update and fix this, need someway to check which dfs actually exits
    logging.info(f"===== Invalid CVX Codes Found: {(len(bad_cvx_codes_df))} =====")
    logging.info(f"===== Invalid CPT Codes Found: {(len(bad_cpt_codes_df))} =====")
    '''
    #############################

    # 5. Filter out missing codes
    #############################
    # Filter missing code for exclusion table
    missing_codes = main_df.loc[
        (main_df[check_codes].astype(str).apply(lambda x: x.str.strip()) == '').all(axis=1)
    ]
    missing_codes['Fallout Reason'] = 'Missing/Invalid Code'

    # Remove missing code records in main df
    main_df = main_df.loc[
        ~(
            main_df[check_codes].astype(str).apply(lambda x: x.str.strip())
            == ''
        ).all(axis=1)
    ]

    # Missing LOINCAnswer w/ LOINC Code (intentionally kept commented)
    # main_df = main_df[(main_df['LOINC'].str.strip() != '') & (main_df['LOINCAnswer'].str.strip() != '')]

    logging.info("===== Procedure code filtering complete =====")
    logging.info(f"===== Missing/Invalid code count: {len(missing_codes)} =====")
    summary_data['Missing_Code_Count'] = len(missing_codes)
    print(f'Invalid Code Count: {len(missing_codes)}')

    #############################

    # 5. Filter for missing/invalid DOS
    #############################
    # Minimum Valid Date
    min_date = pd.to_datetime('1909-01-01')
    # Max Valid Date
    max_date = pd.to_datetime('today').normalize()

    # Filter missing DOS for exclusion table
    missing_dos = main_df.loc[main_df['DOS'].astype('string').str.strip() == ''].copy()
    missing_dos['Fallout Reason'] = 'Missing/Invalid DOS'

    # Remove missing DOS records in main df
    main_df = main_df[
        (main_df['DOS'].astype('string').str.strip() != '')
        & (main_df['DOS'] >= min_date)
        & (main_df['DOS'] <= max_date)
    ]

    print(f'Invalid DOS Count: {len(missing_dos)}')
    logging.info("===== DOS filtering complete =====")
    logging.info(f"===== Bad DOS count: {len(missing_dos)} =====")
    summary_data['Missing_DOS_Count'] = len(missing_dos)

    #############################

    # TX Custom Logic
    #############################
    if source_name == "TX_ImmTrac":
        home_directory = config['output']['output_path']
        ref_file = f"{home_directory}/0_Refs/reference_id.txt"
        populate_reference_id(main_df, ref_file)

    #############################

    # Uniquify Alt IDs
    #############################
    try:
        uniquify_flag = config['custom_logic']['uniquify_alt_id']
    except Exception:
        uniquify_flag = False

    if uniquify_flag:
        print('running uniquify_alt_id_function')
        main_df = uniquify_alt_ids(main_df)
    #############################

    # Remap PCP_Ind
    #############################
    # As per Angel PCP_Ind must be Y or N, remapping from 0 and 1
    # if PCPFlag missing, defaulting to N
    main_df['PCPFlag'] = main_df['PCPFlag'].replace(['0', '', '1'], ['N', 'N', 'Y'])

    #############################

    # Df info for debugging
    print(f'DF length: {len(main_df)}')
    print(f'Data types: {main_df.dtypes}')
    print(f'Columns {main_df.columns}')
    show_memory_usage(main_df)

    # Inovalon ECDS Formatting
    #############################
    # RXProvider / SuppSource for Imm
    main_df.replace({'RxProviderFlag': {'': 'N'}}, inplace=True)
    main_df.replace({'SuppSource': {'': 'R'}}, inplace=True)

    # Create final df in ecds layout
    inovalon_format = pd.DataFrame()

    inovalon_format['MemberKey'] = main_df['MemberKey']
    inovalon_format['ProviderKey'] = main_df['ProviderKey']
    inovalon_format['ReferenceID'] = main_df['ReferenceID']
    inovalon_format['DOS'] = main_df['DOS']
    inovalon_format['DOSThru'] = main_df['DOS']  # Update if needed
    inovalon_format['ProviderType'] = main_df['ProviderType']
    inovalon_format['ProviderTaxonomy'] = main_df['ProviderTaxonomy']
    inovalon_format['CPTPx'] = main_df['CPTPx']
    inovalon_format['HCPCSPx'] = main_df['HCPCSPx']
    inovalon_format['LOINC'] = main_df['LOINC']
    inovalon_format['SNOMED'] = main_df['SNOMED']
    inovalon_format['ICDDX'] = main_df['ICDDX']
    inovalon_format['ICDDX10'] = main_df['ICDDX10']
    inovalon_format['RxNorm'] = main_df['RxNorm']
    inovalon_format['CVX'] = main_df['CVX']
    inovalon_format['Modifier'] = main_df['Modifier']
    inovalon_format['RxProviderFlag'] = main_df['RxProviderFlag']  # Hard coded 'N'
    inovalon_format['PCPFlag'] = main_df['PCPFlag']
    inovalon_format['QuantityDispensed'] = main_df['QuantityDispensed']
    inovalon_format['ICDPx'] = main_df['ICDPx']
    inovalon_format['ICDPx10'] = main_df['ICDPx10']
    inovalon_format['SuppSource'] = main_df['SuppSource']  # Hard coded 'R'
    inovalon_format['Result'] = main_df['Result']
    inovalon_format['Ecdsaltid_00'] = main_df['Ecdsaltid_00']
    inovalon_format['CPTmod'] = main_df['CPTmod']
    inovalon_format['HCPCSmod'] = main_df['HCPCSmod']
    inovalon_format['LOINCAnswer'] = main_df['LOINCAnswer']

    column_order = [
        'MemberKey', 'ProviderKey', 'ReferenceID', 'DOS', 'DOSThru', 'ProviderType',
        'ProviderTaxonomy', 'CPTPx', 'HCPCSPx', 'LOINC', 'SNOMED', 'ICDDX', 'ICDDX10', 'RxNorm',
        'CVX', 'Modifier', 'RxProviderFlag', 'PCPFlag', 'QuantityDispensed', 'ICDPx', 'ICDPx10',
        'SuppSource', 'Result', 'Ecdsaltid_00', 'CPTmod', 'HCPCSmod', 'LOINCAnswer',
    ]

    # CVX Formatting (final layout requires 3 digits)
    mask = inovalon_format['CVX'] != ''
    inovalon_format.loc[mask, 'CVX'] = (
        inovalon_format.loc[mask, 'CVX'].astype(str).str.zfill(3)
    )

    # Results Formatting (preserve numeric precision if numeric, else leave as-is)
    original_result = inovalon_format['Result'].copy()
    tmp_numeric = pd.to_numeric(original_result, errors='coerce').map("{:.10f}".format)
    inovalon_format['Result'] = tmp_numeric.fillna(original_result)
    inovalon_format['Result'] = inovalon_format['Result'].replace("nan", "")

    # ICDDX10 Formatting: strip periods (per original logic)
    inovalon_format['ICDDX10'] = inovalon_format['ICDDX10'].str.replace(r'\.', '', regex=True)

    # adjusting max_colwidth to display all decimals
    pd.set_option('display.max_colwidth', None)

    # Set column order, drop unused columns
    inovalon_format = inovalon_format[column_order]

    # Testing dedupe on final df
    # commented out above due to IL and FL imm dedupe cols
    inovalon_format.fillna('', inplace=True)
    print(f'len before dedupe: {len(inovalon_format)}')
    if source_name in ("FL_ImmResponse", "IL_Icare"):
        inovalon_format = inovalon_format.drop_duplicates(keep='first')
    else:
        dedupe_columns = config['filter_rules']['dedupe_columns']
        idx = dedupe_columns.index('MedicaidID')
        dedupe_columns[idx] = 'MemberKey'
        inovalon_format = inovalon_format.drop_duplicates(
            keep='first', subset=dedupe_columns
        )

    print(f'len after dedupe: {len(inovalon_format)}')

    # Concat all exclusions:
    # New testing 06/27/2025
    if flatfile == '':
        exclusions_final = pd.concat(
            [non_member_match, missing_codes, missing_dos, duplicates], axis=0
        )
    else:
        exclusions_final = pd.concat([non_member_match, missing_codes, missing_dos], axis=0)
 
    # Concat invalid codes (currently only CVX per original logic)
    #invalid_codes = pd.concat([bad_cvx_codes_df])

    # Logging
    logging.info(f"===== Total fallout count: {len(exclusions_final)} =====")
    logging.info(f"===== Total output count: {len(inovalon_format)} =====")

    # Summary DF
    summary_data['Source_Name'] = source_name
    summary_data['Transformation_Date'] = date.today()
    summary_data['Total_Fallout_Count'] = len(exclusions_final)
    summary_data['Total_Output_Count'] = len(inovalon_format)
    summary_data['Status'] = 'Under Review'
    summary_data['Transformer'] = 'Medicaid ETL Engine'
    summary_data['MBU_FileIds'] = inovalon_format['ReferenceID'].unique()

    # Exporting
    #############################
    source_name = config['source_id']
    home_directory = config['output']['output_path']

    # Summary DF
    summary_data['Home_Directory'] = home_directory
    summary_data['Output_Directory'] = f'{home_directory}/1_Final_Output/'

    if date_range:
        start_date, end_date = date_range
        summary_data['Output_FileName'] = (
            f'{source_name}_{datetime.now():%y%m%d}_{start_date}_{end_date}_{mode}_ETL.txt'
        )
        summary_df = pd.DataFrame([summary_data])

        # Files sent to Inovalon will be tab delimited
        # Prod loc: \\MCDQltyPerfRptng\MCDQltyPerfRptng\New Supp Data Drop Location\For Inovalon  # need access!!!
        inovalon_format.to_csv(
            f'{home_directory}/1_Final_Output/{source_name}_{datetime.now():%y%m%d}_{start_date}_{end_date}_{mode}_ETL.txt',
            sep='\t',
            index=False,
        )

        # Export exclusion flatfile
        exclusions_final.to_csv(
            f'{home_directory}/2_Fallout_Report/{source_name}_FallOut_{datetime.now():%y%m%d}_{start_date}_{end_date}.csv',
            index=False,
        )
        invalid_codes_df.to_csv(
            f'{home_directory}/2_Fallout_Report/{source_name}_Invalid_Codes_{datetime.now():%y%m%d}_{start_date}_{end_date}.csv',
            index=False,
        )
        codes_removed_df.to_csv(
            f'{home_directory}/2_Fallout_Report/{source_name}_Codes_Removed_Summary_{datetime.now():%y%m%d}_{start_date}_{end_date}.csv',
            index=False,
        )
        # Export to excel for manual analysis
        inovalon_format.to_csv(
            f'{home_directory}/4_Manual_Review/{source_name}_{datetime.now():%y%m%d}_{start_date}_{end_date}.csv',
            index=False,
        )

        # Export summary data
        summary_df.to_csv(
            f'{home_directory}/5_Summary/{source_name}_Summary_{datetime.now():%y%m%d}_{start_date}_{end_date}.csv',
            sep='\t',
            index=False,
        )
        summary_fileName = (
            f'{source_name}_Summary_{datetime.now():%y%m%d}_{start_date}_{end_date}.csv'
        )

    else:
        summary_data['Output_FileName'] = (
            f'{source_name}_{datetime.now():%y%m%d}_{mode}_ETL.txt'
        )
        summary_df = pd.DataFrame([summary_data])
        # Files sent to Inovalon will be tab delimited
        # Prod loc: \\MCDQltyPerfRptng\MCDQltyPerfRptng\New Supp Data Drop Location\For Inovalon  # need access!!!
        inovalon_format.to_csv(
            f'{home_directory}/1_Final_Output/{source_name}_{datetime.now():%y%m%d}_{mode}_ETL.txt',
            sep='\t',
            index=False,
        )

        # Export exclusion flatfile
        exclusions_final.to_csv(
            f'{home_directory}/2_Fallout_Report/{source_name}_FallOut_{datetime.now():%y%m%d}.csv',
            index=False,
        )
        invalid_codes.to_csv(
            f'{home_directory}/2_Fallout_Report/{source_name}_Invalid_Codes_{datetime.now():%y%m%d}.csv',
            index=False,
        )

        # Export to excel for manual analysis (intentionally commented in original)
        # inovalon_format.to_csv(f'{home_directory}/4_Manual_Review/{source_name}_{datetime.now():%y%m%d}.csv', index=False)

        # Export summary data
        summary_df.to_csv(
            f'{home_directory}/5_Summary/{source_name}_Summary_{datetime.now():%y%m%d}.txt',
            sep='\t',
            index=False,
        )
        summary_fileName = f"{source_name}_Summary_{datetime.now():%y%m%d}.txt"

    if mode in ("data_refresh", "previous_month"):
        summary_path = f'{home_directory}/5_Summary/'
        quick_base_automation.CreateRecord(summary_path, summary_fileName)

        # Exporting to inovalon staging
        inovalon_format.to_csv(
            f'../Inovalon_Staging/{source_name}_{datetime.now():%y%m%d}_{mode}_ETL.txt',
            sep='\t',
            index=False,
        )

    #############################


def main():
    parser = argparse.ArgumentParser(
        description="Run ETL for a specific config and mode."
    )
    parser.add_argument(
        '--config',
        required=True,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        '--mode',
        choices=['data_refresh', 'previous_month', 'sample'],
        required=True,
        help=(
            "Pull last 4 years (datarefresh). Pull previous month (previous_month). "
            "Pull sample top 10000 rows for testing (sample)"
        ),
    )
    parser.add_argument(
        '--date_range',
        nargs=2,
        metavar=('START_DATE', 'END_DATE'),
        help=(
            "Optional: Provide two dates (YYYY-MM-DD YYYY-MM_DD) to override the default "
            "date range in datarefresh mode."
        ),
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Configure logging
    source_name = config['source_id']
    home_directory = config['output']['output_path']
    logging.basicConfig(
        filename=f'{home_directory}/3_Log/{source_name}_log-{datetime.now():%y%m%d}.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )

    # Start timer
    start_time = time.time()
    print('Starting Script...')

    # Run ETL
    run_etl(config, args.mode, args.date_range)

    # Stop timer
    end_time = time.time()
    print(f"Finished in: {round(end_time-start_time)} seconds....")
    logging.info(
        f"===== Processing Complete in: {round(end_time-start_time)} seconds ====="
    )


if __name__ == "__main__":
    main()
