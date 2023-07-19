import jsonlines
from datasets import load_dataset
from evaluate import load
import os
from tqdm import tqdm
from string import Template
import json

os.environ["HF_ALLOW_CODE_EVAL"] = "1"
# 기존 template
CANDIDATE_TEMPLATE = Template(
"""
import sys
from io import StringIO

completion = \"\"\"${COMPLETION}\"\"\"

def ENTRY_POINT(input_str):
    stdin = StringIO(input_str)
    stdout = StringIO()

    sys.stdin = stdin
    sys.stdout = stdout
    exec(
        completion,
        {
            __name__: "__main__",
            "sys": sys,
            "stdin": sys.stdin,
            "stdout": sys.stdout,
        },
    )

    return stdout.getvalue()
"""
    )
# 기존 template
TEST_TEMPLATE = Template("""assert ENTRY_POINT('''${INPUT}''') == '''${OUTPUT}'''""")


solution = load_dataset('deepmind/code_contests')['valid']

# test case를 dataset에서 가져와서 dict로 만들어 task_id 별로 input과 output list를 만들어주었습니다.
# 진행을 하면, {'input': ['2\n17\n5\n','2\n17\n5\n', ... 'output' : ... 이렇게 됩니다.}
test_dicts = {}
for task_id in range(len(solution)):
    test_dicts[task_id] = {'input' : [], 'output':[]}
    for test_type in ["public_tests", "private_tests", "generated_tests"]:
        input_seq = solution[task_id][test_type]['input']
        output_seq = solution[task_id][test_type]['output']
        test_dicts[task_id]["input"].extend(input_seq)
        test_dicts[task_id]["output"].extend(output_seq)        

pred_jsonl_file = './pred/first_gen_copy.jsonl'
code_eval = load('code_eval')

# 에러로 인해 하나의 sample에서만 진행하고 있습니다.
# completion = solution[4]['solutions']['solution'][10]는 일부러 정답 코드를 test 해보려고 작성했습니다.
with open('./pred/first_gen_eval.jsonl', encoding= "utf-8",mode="w") as writer:        
    with jsonlines.open(pred_jsonl_file) as pred_reader:
        for pred_item in pred_reader:
            epoch = int(pred_item['epoch'])
            task_id = int(pred_item['task_id'])
            prompt = pred_item['prompt']
            # completion = pred_item['completion'].replace('\\n', '\n')
            completion = solution[4]['solutions']['solution'][10]
            
            # 해당 함수는 python 2.7로 생성된 함수를 python 3.0으로 전환해주는 함수입니다.
            completion = refactor_code(completion)
            
            # 여기서 대체를 진행합니다.
            candidate = CANDIDATE_TEMPLATE.substitute(
                        {"COMPLETION": completion}
                    )
            # 이래는 ["assert ENTRY_POINT('''2\n17\n5\n''') == '''2 16\n2 4\n'''", ]로 정상적으로 진행됩니다.
            test_case = []
            for idx in range(len(test_dicts[task_id]['input'])):                
                input_seq, output_seq = test_dicts[task_id]['input'][idx].replace('\\n', '\n'), test_dicts[task_id]['output'][idx].replace('\\n', '\n')
                test_case.append(TEST_TEMPLATE.substitute(
                    {"INPUT": input_seq, "OUTPUT": output_seq}
                ))
            # pass_at_k, results = code_eval.compute(references=test_case, predictions=[[candidate] for _ in range(len(test_case))])
            
            # 단일 케이스에 대해서만 진행해주기 위해 이런 식으로 했습니다.
            pass_at_k, results = code_eval.compute(references=[test_case[0]], predictions=[[candidate]])
            line = {'epoch' : str(epoch), 'task_id' : str(task_id), 'completion' : completion.replace('\n', '\\n'), 'prompt' : prompt, 'pass_at_1' : pass_at_k['pass@1']}
            writer.write(json.dumps(line) + "\n")