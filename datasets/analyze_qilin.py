'''
@ref: Qilin: A Multimodal Information Retrieval Dataset with APP-level User Sessions
@author: Jia Chen, Qian Dong, Haitao Li, Xiaohui He, Yan Gao, Shaosheng Cao, Yi Wu, Ping Yang, Chen Xu, Yao Hu, Qingyao Ai, Yiqun Liu.
'''

import numpy as np
from datasets import *
import pandas as pd
import json


def cal_max_depth(x):
    x = json.loads(x)
    max_depth = -1
    for e in x:
        if e['position'] > max_depth:
            max_depth = e['position']
    return max_depth


def cal_first_click_pos(x):
    x = json.loads(x)
    first_click_pos = np.nan
    is_clicked = False
    for e in x:
        if e['click'] > 0:
            if is_clicked:
                first_click_pos = min(first_click_pos, e['position'])
            else:
                first_click_pos = e['position']
                is_clicked = True
    return first_click_pos


def get_engage_data(x):
    x = json.loads(x)
    imp = len(x)
    click, like, collect, follow, comment = 0, 0, 0, 0, 0
    for e in x:
        if e['click'] > 0:
            click += 1
            if e['like'] > 0:
                like += 1
            if e['collect'] > 0:
                collect += 1
            if e['share'] > 0:
                follow += 1
            if e['comment'] > 0:
                comment += 1
    return [imp, click, like, collect, follow, comment]


def cal_click_browse(data, info):
    n = data.shape[0]

    search_result_details = data[f'{info}_result_details_with_idx']
    avg_browsing_depth = search_result_details.apply(cal_max_depth).mean()
    avg_first_click_pos = search_result_details.apply(cal_first_click_pos).mean()

    engages = search_result_details.apply(get_engage_data)
    ctr = engages.apply(lambda x: x[1]).sum() / engages.apply(lambda x: x[0]).sum()
    avg_click_num = engages.apply(lambda x: x[1]).mean()

    like_rate = engages.apply(lambda x: x[2]).sum() / engages.apply(lambda x: x[1]).sum()
    collect_rate = engages.apply(lambda x: x[3]).sum() / engages.apply(lambda x: x[1]).sum()
    share_rate = engages.apply(lambda x: x[4]).sum() / engages.apply(lambda x: x[1]).sum()
    comment_rate = engages.apply(lambda x: x[5]).sum() / engages.apply(lambda x: x[1]).sum()

    print(f'{info} n={n}')
    print(f'avg_browsing_depth={avg_browsing_depth}')
    print(f'avg_first_click_pos={avg_first_click_pos}')
    print(f'avg_click_num={avg_click_num}')

    print(f'ctr={ctr}')
    print(f'like_rate={like_rate}')
    print(f'collect_rate={collect_rate}')
    print(f'share_rate={share_rate}')
    print(f'comment_rate={comment_rate}\n')


def cal_ctr_position(data, info):
    data = data.sort_values(by='begin_time')
    fea_prefix = 'search' if info == 'search' else 'rec'
    fea_name = f'{fea_prefix}_result_details_with_idx'
    result_details = data[fea_name].tolist()
    session_idxs = data['session_idx'].tolist()
    session_dict, session_position_dict, position_dict = {}, {}, {}
    for idx, result_detail in enumerate(result_details):
        detail = json.loads(result_detail)
        session_idx = session_idxs[idx]

        if session_idx not in session_dict:
            session_dict[session_idx] = []
        session_dict[session_idx].append(1)
        session_position = len(session_dict[session_idx])
        session_position = session_position if session_position <= 10 else 11
        if session_position not in session_position_dict:
            session_position_dict[session_position] = [0, 0]

        for note in detail:
            position, click = note['position'], note['click']
            if position <= 10:
                if position not in position_dict:
                    position_dict[position] = [0, 0]
            else:
                position = 11
                if position not in position_dict:
                    position_dict[position] = [0, 0]
            position_dict[position][0] += 1
            position_dict[position][1] += note['click']
            session_position_dict[session_position][0] += 1
            session_position_dict[session_position][1] += note['click']

    print(f'{info}')
    print('position bias')
    position_keys = sorted(position_dict.keys())
    session_position_keys = sorted(session_position_dict.keys())
    for key in position_keys:
        print(f'{key} imp={position_dict[key][0]} click={position_dict[key][1]} ctr={position_dict[key][1]/position_dict[key][0]}')

    print('session bias')
    for key in session_position_keys:
        print(f'{key} imp={session_position_dict[key][0]} click={session_position_dict[key][1]} ctr={session_position_dict[key][1]/session_position_dict[key][0]}')

    print('\n')


def process_arrow_to_csv_rec(e):
    # 1. trim into int
    for key in ['rec_result_details_with_idx']:
        for sub_e in e[key]:
            for subkey in ['position', 'click', 'like', 'comment', 'collect', 'share', 'request_timestamp']:
                sub_e[subkey] = int(sub_e[subkey])

            if np.isnan(sub_e['page_time']) or sub_e['page_time'] < 0.0:
                sub_e['page_time'] = -1
        e[key] = sorted(e[key], key=lambda x: x['request_timestamp'])

    # 2. list to json_str
    for key in ['recent_clicked_note_idxs', 'rec_result_details_with_idx']:
        e[key] = json.dumps((e[key]))

    return e


def process_arrow_to_csv_search(e):
    # 1. trim into int
    for key in ['query_from_type', 'search_result_details_with_idx']:
        if key == 'search_result_details_with_idx':
            for sub_e in e[key]:
                for subkey in ['position', 'click', 'like', 'comment', 'collect', 'share', 'search_timestamp']:
                    sub_e[subkey] = int(sub_e[subkey])
                if np.isnan(sub_e['page_time']) or sub_e['page_time'] == -1.0:
                    sub_e['page_time'] = -1
            e[key] = sorted(e[key], key=lambda x: x['search_timestamp'])
        else:
            if e[key] is not None:
                e[key] = int(e[key])

    # 2. list to json_str
    for key in ['recent_clicked_note_idxs', 'search_result_details_with_idx']:
        e[key] = json.dumps((e[key]))
    return e


def process_arrow_to_csv_dqa(e):
    # 1. trim into int
    for key in ['query_from_type', 'is_like_clk', 'is_onebox_trace_clk', 'is_content_clk', 'is_experience_clk',
                'search_result_details_with_idx']:
        if key == 'search_result_details_with_idx':
            for sub_e in e[key]:
                for subkey in ['position', 'click', 'like', 'comment', 'collect', 'share', 'search_timestamp']:
                    sub_e[subkey] = int(sub_e[subkey])
                if np.isnan(sub_e['page_time']) or sub_e['page_time'] == -1.0:
                    sub_e['page_time'] = -1
            e[key] = sorted(e[key], key=lambda x: x['search_timestamp'])
        else:
            if e[key] is not None:
                e[key] = int(e[key])

    # 2. list to json_str
    for key in ['recent_clicked_note_idxs', 'search_result_details_with_idx', 'ref_note_idx_list']:
        if e[key] is not None:
            e[key] = json.dumps((e[key]))
        else:
            e[key] = json.dumps(([]))
    return e


def get_first_result_time(x):
    return json.loads(x)[0]['search_timestamp']


def get_first_result_time_rec(x):
    return json.loads(x)[0]['request_timestamp']


def remove_cols(hf_data, cols):
    for col in cols:
        if col in hf_data.column_names:
            hf_data = hf_data.remove_columns([col])
    return hf_data

def arrow_to_csv_final(base_directory):
    data = load_from_disk(base_directory)
    to_removed = ['bm25_results', 'dpr_results']
    search_1 = remove_cols(data['search_train'], to_removed)
    search_2 = remove_cols(data['search_test'], to_removed)
    rec_1 = remove_cols(data['recommendation_train'], to_removed+['query', 'rec_results'])
    rec_2 = remove_cols(data['recommendation_test'], to_removed+['query', 'rec_results'])
    dqa_data = remove_cols(data['dqa'], to_removed)

    search_1 = search_1.map(process_arrow_to_csv_search)
    search_2 = search_2.map(process_arrow_to_csv_search)
    rec_1 = rec_1.map(process_arrow_to_csv_rec)
    rec_2 = rec_2.map(process_arrow_to_csv_rec)
    dqa_data = dqa_data.map(process_arrow_to_csv_dqa)

    # save as dataframe
    search_data_df = pd.concat([search_1.to_pandas(), search_2.to_pandas()], axis=0, ignore_index=True)
    rec_data_df = pd.concat([rec_1.to_pandas(), rec_2.to_pandas()], axis=0, ignore_index=True)
    dqa_data_df = dqa_data.to_pandas()

    # dealing with invalid values: -1, null, NaN
    search_data_df = search_data_df.fillna(-1)
    search_data_df = search_data_df.replace(r'^\s*$', -1, regex=True)  # 处理空字符串

    rec_data_df = rec_data_df.fillna(-1)
    rec_data_df = rec_data_df.replace(r'^\s*$', -1, regex=True)  # 处理空字符串

    dqa_data_df = dqa_data_df.fillna(-1)
    dqa_data_df = dqa_data_df.replace(r'^\s*$', -1, regex=True)  # 处理空字符串

    search_data_df = search_data_df.sort_values(by=['session_idx'])
    search_data_df = search_data_df.loc[
        search_data_df['search_result_details_with_idx'].apply(get_first_result_time).sort_values().index]

    rec_data_df = rec_data_df.sort_values(by=['session_idx'])
    rec_data_df = rec_data_df.loc[
        rec_data_df['rec_result_details_with_idx'].apply(get_first_result_time_rec).sort_values().index]

    dqa_data_df = dqa_data_df.sort_values(by=['session_idx'])
    dqa_data_df = dqa_data_df.loc[
        dqa_data_df['search_result_details_with_idx'].apply(get_first_result_time).sort_values().index]

    search_data_df.to_csv(f'datasets/toy_data/csv/search_toy.csv', index=False)
    rec_data_df.to_csv(f'datasets/toy_data/csv/recommendation_toy.csv', index=False)
    dqa_data_df.to_csv(f'datasets/toy_data/csv/dqa_toy.csv', index=False)


def cal_transition_rate(data):
    data = data.sort_values(by='begin_time')
    num = data.shape[0]
    print(f'All request num={num}')
    user_session_dict = {}
    for _, row in data.iterrows():
        session_idx, user_idx, begin_time, engage_data, \
            request_type = row['session_idx'], row['user_idx'], row['begin_time'], row['engage_data'], row['type']
        if user_idx not in user_session_dict:
            user_session_dict[user_idx] = {}
        if len(user_session_dict[user_idx]) == 0:
            user_session_dict[user_idx] = []
            user_session_dict[user_idx].append([[begin_time, engage_data, request_type]])
            continue
        if (begin_time - user_session_dict[user_idx][-1][-1][0] <= 1800):  # add new request into existing session
            user_session_dict[user_idx][-1].append([begin_time, engage_data, request_type])
        else:  # add new session
            user_session_dict[user_idx].append([[begin_time, engage_data, request_type]])
    print(len(user_session_dict))

    transition_dict = [0, 0, 0, 0, 0]
    transition_engage_dict = {'S->S': [0, 0, []], 'R->S': [0, 0, []],
                              'R->R': [0, 0, []], 'S->R': [0, 0, []]}
    for key in user_session_dict:
        sessions = user_session_dict[key]
        for session in sessions:
            if len(session) > 1:
                for i in range(len(session) - 1):
                    x, y = session[i][-1], session[i+1][-1]
                    transition_dict[0] += 1
                    if x == 'S' and y == 'S':
                        transition_dict[1] += 1
                        transition_engage_dict['S->S'][0] += session[i + 1][1][0]
                        transition_engage_dict['S->S'][1] += session[i + 1][1][1]
                        transition_engage_dict['S->S'][2].append(session[i + 1][1][1])
                    elif x == 'R' and y == 'S':
                        transition_dict[2] += 1
                        transition_engage_dict['R->S'][0] += session[i + 1][1][0]
                        transition_engage_dict['R->S'][1] += session[i + 1][1][1]
                        transition_engage_dict['R->S'][2].append(session[i + 1][1][1])
                    elif x == 'R' and y == 'R':
                        transition_dict[3] += 1
                        transition_engage_dict['R->R'][0] += session[i + 1][1][0]
                        transition_engage_dict['R->R'][1] += session[i + 1][1][1]
                        transition_engage_dict['R->R'][2].append(session[i + 1][1][1])
                    else:
                        transition_dict[4] += 1
                        transition_engage_dict['S->R'][0] += session[i + 1][1][0]
                        transition_engage_dict['S->R'][1] += session[i + 1][1][1]
                        transition_engage_dict['S->R'][2].append(session[i + 1][1][1])

    print(transition_dict)
    print(f'S->S: {transition_dict[1]/transition_dict[0]}')
    print(f'R->S: {transition_dict[2] / transition_dict[0]}')
    print(f'R->R: {transition_dict[3] / transition_dict[0]}')
    print(f'S->R: {transition_dict[4] / transition_dict[0]}')

    for key in transition_engage_dict:
        if transition_engage_dict[key][0] != 0:
            print(f'{key}: ctr={transition_engage_dict[key][1]/transition_engage_dict[key][0]} avg_click_num={np.array(transition_engage_dict[key][2]).mean()}')
        else:
            print('NA')

def cal_query_analysis(data):
    data = data.sort_values(by='begin_time')
    user_session_dict = {}
    query_from_type_dict = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    query_length_dict = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    query_length_engage_dict = {}
    for _, row in data.iterrows():
        session_idx, user_idx, begin_time, engage_data, query, query_from_type = \
            row['session_idx'], row['user_idx'], row['begin_time'], row['engage_data'], row['query'], row['query_from_type']
        query_from_type_dict[0] += 1
        query_from_type_dict[int(query_from_type)] += 1

        query_length_dict[0] += 1
        q_len = len(query) if len(query) > 3 else 3
        q_len = q_len - 2 if q_len <= 10 else 9
        if q_len > 0:
            query_length_dict[q_len] += 1

        if q_len not in query_length_engage_dict:
            query_length_engage_dict[q_len] = [0, 0, []]
        query_length_engage_dict[q_len][0] += engage_data[0]
        query_length_engage_dict[q_len][1] += engage_data[1]
        query_length_engage_dict[q_len][2].append(engage_data[1])

        if user_idx not in user_session_dict:
            user_session_dict[user_idx] = {}
        if len(user_session_dict[user_idx]) == 0:
            user_session_dict[user_idx] = []
            user_session_dict[user_idx].append([[begin_time, engage_data, query]])
            continue
        if (begin_time - user_session_dict[user_idx][-1][-1][0] <= 1800):  # add new request into existing session
            user_session_dict[user_idx][-1].append([begin_time, engage_data, query])
        else:  # add new session
            user_session_dict[user_idx].append([[begin_time, engage_data, query]])

    # query source
    print(f'query source, n={query_from_type_dict[0]}')
    for i in range(1, 9):
        print(f'{i} {query_from_type_dict[i]/query_from_type_dict[0]}')
    print('\n')

    # query length
    print(f'query length, n={query_length_dict[0]}')
    for i in range(1, 10):
        print(f'{i} {query_length_dict[i]/query_length_dict[0] * 100}% ctr={query_length_engage_dict[i][1]/query_length_engage_dict[i][0]} '
              f'avg_click_num={np.array(query_length_engage_dict[i][2]).mean()}')
    print('\n')

    # query reformulation, add, delete, change, repeat, others
    reformulation_dict = [0, 0, 0, 0, 0, 0]
    examples = [[], [], [], [], []]
    for key in user_session_dict:
        sessions = user_session_dict[key]
        for session in sessions:
            if len(session) > 1:
                for i in range(len(session) - 1):
                    q1_raw, q2_raw = session[i][-1], session[i+1][-1]
                    q1, q2 = list(q1_raw), list(q2_raw)
                    reformulation_dict[0] += 1

                    q_plus, q_minus, q_intersect = [], [], []
                    for w in q1:
                        if w in q2:
                            q_intersect.append(w)
                        else:
                            q_minus.append(w)
                    for w in q2:
                        if w in q1:
                            if w not in q_intersect:
                                q_intersect.append(w)
                        else:
                            q_plus.append(w)

                    if len(q_plus) > 0 and len(q_minus) == 0:
                        reformulation_dict[1] += 1
                        examples[0].append([q1_raw, q2_raw])
                    elif len(q_plus) == 0 and len(q_minus) > 0:
                        reformulation_dict[2] += 1
                        examples[1].append([q1_raw, q2_raw])
                    elif len(q_plus) > 0 and len(q_minus) > 0 and len(q_intersect) > 0:
                        reformulation_dict[3] += 1
                        examples[2].append([q1_raw, q2_raw])
                    elif len(q_plus) == 0 and len(q_minus) == 0 and len(q_intersect) > 0:
                        reformulation_dict[4] += 1
                        examples[3].append([q1_raw, q2_raw])
                    elif len(q_plus) > 0 and len(q_minus) > 0 and len(q_intersect) == 0:
                        reformulation_dict[5] += 1
                        examples[4].append([q1_raw, q2_raw])
    print(f'query reformulation, n={reformulation_dict[0]}')
    reformulation_name = ['Add', 'Delete', 'Change', 'Repeat', 'Others']
    for i in range(1, 6):
        print(f'{reformulation_name[i-1]} {reformulation_dict[i] / reformulation_dict[0]}')
        print(examples[i-1][:20])
    print('\n')


def cal_hetero_results(data, note_type_dict, info):
    result_details = data[f'{info}_result_details_with_idx'].tolist()
    image_imp_dict, video_imp_dict = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    image_click_dict, video_click_dict = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for detail in result_details:
        detail_obj = json.loads(detail)
        for result in detail_obj:
            click, note_id, position = result['click'], result['note_id'], result['position']
            if note_id in note_type_dict:
                note_type = note_type_dict[note_id]
            else:
                continue
            if position > 10:
                position = 11
            if note_type == 1:
                image_imp_dict[0] += 1
                image_imp_dict[int(position)] += 1
                image_click_dict[0] += click
                image_click_dict[int(position)] += click
            else:
                video_imp_dict[0] += 1
                video_imp_dict[int(position)] += 1
                video_click_dict[0] += click
                video_click_dict[int(position)] += click

    print(f'Scenario: {info}')
    # exposure
    print('Overall distribution')
    print(f'image-text={image_imp_dict[0]/(image_imp_dict[0]+video_imp_dict[0])} '
          f'video={video_imp_dict[0]/(image_imp_dict[0]+video_imp_dict[0])}')
    print('Distribution at Rank')
    for i in range(1, 12):
        print(
            f'rank={i} image-text={image_imp_dict[i] / (image_imp_dict[i] + video_imp_dict[i])} video={video_imp_dict[i] / (image_imp_dict[i] + video_imp_dict[i])}')

    # ctr
    print('Overall CTR')
    print(
        f'image-text={image_click_dict[0] / image_imp_dict[0]} video={video_click_dict[0] / video_imp_dict[0]}')
    print('CTR at Rank')
    for i in range(1, 12):
        print(
            f'rank={i} image-text={image_click_dict[i] / image_imp_dict[i]} video={video_click_dict[i] / video_imp_dict[i]}')


def analyze_qilin(base_directory):

    search_data = pd.read_csv(f'{base_directory}/search_toy.csv')
    rec_data = pd.read_csv(f'{base_directory}/recommendation_toy.csv')
    dqa_data = pd.read_csv(f'{base_directory}/dqa_toy.csv')

    # 1. search & rec
    print('1. Print search and recommendation click and browses...')
    search_final_data = pd.concat([search_data, dqa_data], axis=0, ignore_index=True)
    cal_click_browse(search_final_data, 'search')
    cal_click_browse(rec_data, 'rec')

    # 2. search & dqa
    print('2. Print search with and without dqa...')
    dqa_on = dqa_data[dqa_data.is_like_clk >= 0]
    dqa_off = dqa_data[dqa_data.is_like_clk < 0]
    search_final_data = pd.concat([search_data, dqa_off], axis=0, ignore_index=True)
    cal_click_browse(search_final_data, 'search')
    cal_click_browse(dqa_on, 'search')

    # 3. ctr with position & session position
    print('3. Print ctr w.r.t. ranking and session positions...')
    search_final_data = pd.concat([search_data, dqa_data], axis=0, ignore_index=True)
    search_final_data['begin_time'] = search_final_data['search_result_details_with_idx'].apply(lambda x: json.loads(x)[0]['search_timestamp'])
    rec_data['begin_time'] = rec_data['rec_result_details_with_idx'].apply(
        lambda x: json.loads(x)[0]['request_timestamp'])
    cal_ctr_position(search_final_data, 'search')
    cal_ctr_position(rec_data, 'rec')

    # 4. transition rate
    print('4. Print transition rate...')
    search_data['begin_time'] = search_data['search_result_details_with_idx'].apply(lambda x: json.loads(x)[0]['search_timestamp'])
    rec_data['begin_time'] = rec_data['rec_result_details_with_idx'].apply(
        lambda x: json.loads(x)[0]['request_timestamp'])
    dqa_data['begin_time'] = dqa_data['search_result_details_with_idx'].apply(
        lambda x: json.loads(x)[0]['search_timestamp'])

    search_data['type'] = 'S'
    rec_data['type'] = 'R'
    dqa_data['type'] = 'S'

    search_data['engage_data'] = search_data['search_result_details_with_idx'].apply(get_engage_data)
    rec_data['engage_data'] = rec_data['rec_result_details_with_idx'].apply(get_engage_data)
    dqa_data['engage_data'] = dqa_data['search_result_details_with_idx'].apply(get_engage_data)

    search_data_ = search_data[['session_idx', 'user_idx', 'begin_time', 'engage_data', 'type']]
    rec_data_ = rec_data[['session_idx', 'user_idx', 'begin_time', 'engage_data', 'type']]
    dqa_data_ = dqa_data[['session_idx', 'user_idx', 'begin_time', 'engage_data', 'type']]

    all_data = pd.concat([search_data_, rec_data_, dqa_data_], axis=0, ignore_index=True)
    cal_transition_rate(all_data)

    # 5. query analysis
    print('5. Print query analysis...')
    search_data['begin_time'] = search_data['search_result_details_with_idx'].apply(lambda x: json.loads(x)[0]['search_timestamp'])
    dqa_data['begin_time'] = dqa_data['search_result_details_with_idx'].apply(
        lambda x: json.loads(x)[0]['search_timestamp'])

    search_data['engage_data'] = search_data['search_result_details_with_idx'].apply(get_engage_data)
    dqa_data['engage_data'] = dqa_data['search_result_details_with_idx'].apply(get_engage_data)

    search_data = search_data[['session_idx', 'user_idx', 'begin_time', 'engage_data', 'query', 'query_from_type']]
    dqa_data = dqa_data[['session_idx', 'user_idx', 'begin_time', 'engage_data', 'query', 'query_from_type']]
    all_search_data = pd.concat([search_data, dqa_data], axis=0, ignore_index=True)
    cal_query_analysis(all_search_data)

    # 6. result distribution analysis, need note features for mapping
    # print('Print result distribution...')
    # names = ['old.csv', 'extra.csv', 'rag.csv', 'dqa.csv']
    # base_directory = 'path/to/note_features'
    # dfs = [pd.read_csv(f'{base_directory}/{name}') for name in names]
    # note_feat = pd.concat(dfs, ignore_index=True)
    # note_ids, note_types = note_feat['note_id'].tolist(), note_feat['note_type'].tolist()
    # note_type_dict = dict(zip(note_ids, note_types))
    # print(f'all note num={len(note_type_dict)}')
    # search_final_data = pd.concat([search_data, dqa_data], axis=0, ignore_index=True)
    # cal_hetero_results(search_final_data, note_type_dict, 'search')
    # cal_hetero_results(rec_data, note_type_dict, 'rec')


if __name__ == '__main__':

    # analyze qilin data, dir=path/to/qilin_data_csv
    # if you only have arrow data, please transform into csv data first
    # here we take ../toy_data/csv as an example
    base_directory = './toy_data/csv'
    print(f'Analyzing {base_directory}...')
    print('Note that this is just a toy script. Please download the whole dataset for analysis.\n')
    analyze_qilin(base_directory)
