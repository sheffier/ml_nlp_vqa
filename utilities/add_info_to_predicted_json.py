import collections
import json
import re
import sys

predictions_path = input('Path to JSON file with predictions (list of {"question_id": , "answer": }): ') or \
                         r'nlvr_OpenEnded_mscoco_test_nlvr_b512_lr_1e-4_real_70000_results.json'

nlvr_json_path = input("Path to NLVR2's JSON file (e.g. test1.json): ") or \
                       r'C:\Users\OR\Documents\Academics\VisualReasoningProject\ml_nlp_vqa\nlvr-test1.json'

with open(predictions_path, 'r') as f:
    j = json.load(f)

with open(nlvr_json_path, 'r') as f:
    nlvr = collections.defaultdict(lambda: {})
    for line in f.readlines():
        q = json.loads(line)
        id = q['identifier']
        nlvr[id]['sentence'] = q['sentence']
        nlvr[id]['gt'] = q['label']

for pred in j:
    id = pred['question_id']
    pred['sentence'] = nlvr[id]['sentence']
    pred['gt'] = nlvr[id]['gt']
    image_id = re.sub(r'-\d$', '', id)
    pred['path'] = r'/vol/scratch/erez/ml_nlp_vqa/snmn/exp_nlvr/nlvr_images/images/test1/{}.png'.format(image_id)

with open('results-with-images-and-questions.json', 'w') as f:
    json.dump(j, f, indent=2)
