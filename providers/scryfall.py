
import json
import pandas as pd
def create_set_csv_from_scryfall_oracle_json(oracle_json_path,
                                            set_codes,
                                            output_csv_path):
    '''
        From oracle bulk data file:
        https://scryfall.com/docs/api/bulk-data
        See 'Default Cards' json:
        https://c2.scryfall.com/file/scryfall-bulk/default-cards/default-cards-20210513210311.json
    '''
    oracle_json = json.load(open(oracle_json_path,'r'))
    set_codes = set([x.lower() for x in set_codes])
    set_cards = []
    card_names = set()
    for obj in oracle_json:
        if obj['set'].lower() in set_codes:
            if 'card_faces' in obj:
                card = obj['card_faces'][0]
            else: 
                card = obj
            if obj['lang'] == 'ja':
                continue
            if card['name'] in card_names:
                continue
            else: 
                card_names.add(card['name'])
            set_cards.append({'Name':card['name'],
                                'Mana Cost':card['mana_cost'],
                                'Mana Value':obj['cmc'],
                                'Rarity':obj['rarity'],
                                'Set Code':obj['set'],
                                'Type':obj['type_line'],
                                'Image URL':card['image_uris']['normal']})

    set_cards = sorted(set_cards,key=lambda x:x['Name'])

    df = pd.DataFrame.from_records(set_cards)
    df.to_csv(output_csv_path,index=False)