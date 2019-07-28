import json
import re

with open(r'C:\Users\user\Desktop\University\TAU\year1\NLP\project\nlvr\nlvr2\data\dev.json', 'r') as f:
    examples = [json.loads(line) for line in f.readlines() if line]

base = {
    "info": {
    }
}

with open('train2014QuestionsForSNMN.json', 'w') as dq:
    image_questions = {
        "questions": [
            {
                "question_id": example['identifier'],
                "image_id": None,
                "image_name": re.match(r'([-\w]+)-\d', example['identifier']).group(1) + '-0',
                "question": example['sentence']
            }
            for example in examples
        ]
    }
    image_questions.update(base)
    json.dump(image_questions, dq, indent=2)

with open('train2014AnnotationsForSNMN.json', 'w') as da:
    image_annotations = {
        "annotations": [
            {
                "question_id": example['identifier'],
                "image_id": None,
                "answers": [
                    {
                        "answer_id": 0,
                        "answer": "false",
                        # "answer_confidence":
                    },
                    {
                        "answer_id": 1,
                        "answer": "true",
                        # "answer_confidence":
                    }
                ]
            }
            for example in examples
        ]
    }
    image_annotations.update(base)
    json.dump(image_annotations, da, indent=2)
