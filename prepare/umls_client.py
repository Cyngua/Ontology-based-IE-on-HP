'''
given the concept's SNOMED CT id map to CUI and then retrieve other metadata
'''
import os
import requests
import pprint
import tqdm
import pandas as pd
import json
from functools import lru_cache
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv('UMLS_API_KEY')

class UMLSClient:
    BASE_URL = "https://uts-ws.nlm.nih.gov"

    def __init__(self, api_key, version="current"):
        self.api_key = api_key
        self.version = version

    def _make_request(self, endpoint):
        url = f"{self.BASE_URL}/rest/content/{self.version}/{endpoint}"
        params = {"apiKey": self.api_key}
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json().get("result", {})

    @lru_cache(maxsize=10000)
    def get_concept_info_cui(self, cui):
        """
        Retrieve the main concept info: name, semantic types, and URLs to atoms and relations.
        """
        result = self._make_request(f"CUI/{cui}")
        return {
            "cui": cui,
            "name": result.get("name"),
            "semantic_types": [st["name"] for st in result.get("semanticTypes", [])][0],
            "atom_url": result.get("atoms"),
            "relations_url": result.get("relations")
        }

    @lru_cache(maxsize=10000)
    def get_concept_info_snomedct(self, ui):
        '''
        Retrieve the concept information using SNOMED CT code
        '''
        sabs = 'snomedct_us'
        page = 1
        path = '/search/'+self.version
        query = {'apiKey':self.api_key, 'string':ui, 'rootSource':sabs, 'inputType':'sourceUI', 'pageNumber':page}
        output = requests.get(self.BASE_URL+path, params=query)
        outputJson = output.json()
        results = (([outputJson['result']])[0])['results']
        if len(results) == 0:
            print(f'No results found for {ui}')
            return {
                    "cui": None,
                    "name": None,
                    "semantic_types": None,
                    "atoms_url": None,
                    "relations_url": None
                }
        concept_url = results[0]['uri']
        params = {"apiKey": self.api_key}
        response = requests.get(concept_url, params=params)
        response.raise_for_status()
        query = response.json().get("result", {})
        return {
            "cui": results[0].get('ui'),
            "name": query.get("name"),
            "semantic_types": [st["name"] for st in query.get("semanticTypes", [])][0],
            "atom_url": query.get("atoms"),
            "relations_url": query.get("relations")
        }

def get_semantic_types(client: UMLSClient, ui_list):
    """
    Get semantic type names by a list of snomedct ui.
    """
    semantic_map = {}
    result = None
    for ui in tqdm.tqdm(ui_list):
        if ui not in semantic_map.keys():
            result = client.get_concept_info_snomedct(ui)
        if result:
            semantic_map[ui] = result.get("semantic_types", '')
    return semantic_map

def get_concept_info_all(client: UMLSClient, ui_list):
    '''
    Get all the relative concept info given a list of us
    '''
    concept_info_map = {}
    result = None
    for ui in tqdm.tqdm(ui_list):
        result = client.get_concept_info_snomedct(ui)
        concept_info_map[str(ui)] = result
    return concept_info_map


def test():
    client = UMLSClient(api_key)
    ui = 22298006
    info = client.get_concept_info_snomedct(ui)
    pprint.pprint(info)

    ui_list = [
        91936005,
        95563007,
        45595009,
        95563007,
        1835003,
        310244003,
        19387007,
        57653000,
        268910001,
        13467000,
        56783008,
        122865005
    ]
    mapping = get_semantic_types(client, ui_list)
    pprint.pprint(mapping)

def main():
    client = UMLSClient(api_key)
    FOLDER = 'data/benchmark_data'
    annotation_df = pd.read_csv(os.path.join(FOLDER, 'train_annotations.csv'))
    ui_list = annotation_df['concept_id'].unique()
    mapping = get_concept_info_all(client, ui_list)
    # Write to JSON
    with open('concept_info_map.json', 'w') as f:
        json.dump(mapping, f, indent=2)

if __name__ == '__main__':
    main()
