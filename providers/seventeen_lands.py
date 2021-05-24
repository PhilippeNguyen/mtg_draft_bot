import pandas as pd
import numpy as np

def parse_data_csv(csv_path):

    init_df = pd.read_csv(csv_path,nrows=1)
    col_names = init_df.columns
    pack_cols = [ name
                    for name in col_names
                    if name.startswith('pack_card_')]
    pool_cols = [ name
                    for name in col_names
                    if name.startswith('pool_')]
    


    #check to make sure the pool_ and pack_card_ card names match
    pack_names_no_prefix = [name.replace('pack_card_','') for name in pack_cols]
    pool_names_no_prefix = [name.replace('pool_','') for name in pool_cols]
    for pack_name,pool_name in zip(pack_names_no_prefix,
                                pool_names_no_prefix):
        assert pack_name == pool_name

    inv_map = {name:idx for idx,name in enumerate(pack_names_no_prefix)}


    picked_names = []
    pack_tensor = None
    pool_tensor = None
    for data_df in pd.read_csv(csv_path, chunksize=10000):
        if pack_tensor is None:
            pack_tensor = data_df[pack_cols].values.astype(np.uint8)
        else:
            pack_tensor = np.vstack([pack_tensor,
                                    data_df[pack_cols].values.astype(np.uint8)])
        if pool_tensor is None:
            pool_tensor = data_df[pool_cols].values.astype(np.uint8)
        else:
            pool_tensor = np.vstack([pool_tensor,
                                    data_df[pool_cols].values.astype(np.uint8)])
        picked_names.extend(data_df['pick'])


    data = np.concatenate([pool_tensor,pack_tensor],axis=1)
    
    
    target = np.zeros_like(pack_tensor,dtype=np.uint8)
    for n,name in enumerate(picked_names):
        card_idx = inv_map[name]
        target[n,card_idx] = 1

    card_name_df = pd.DataFrame.from_records([{'Name':x} for x in pack_names_no_prefix])
    return data,target,card_name_df


def create_set_csv_from_draft_csv(draft_csv_path,output_csv):
    df = pd.read_csv(draft_csv_path,nrows=1)
    cols = df.columns
    card_names = []
    for col in cols:
        if not col.startswith('pack_card_'):
            continue
        card_name = col[10:] #remove the 'pack_card_' prefix
        card_names.append({'Name':card_name})

    card_names = sorted(card_names,key=lambda x:x['Name'])
    out_df = pd.DataFrame.from_records(card_names)
    out_df.to_csv(output_csv,index=False)

