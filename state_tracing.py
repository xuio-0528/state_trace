import ast
import jsonlines
from tree_sitter import Language, Parser

pred_stmt_trace_list = []
solution_stmt_trace_list = []

Language.build_library(
    # Store the library in the `build` directory
    'build/my-languages.so',
    # Include one or more languages
    [
        './tree-sitter-python'
    ]
)

parser = Parser()
PY_LANGUAGE = Language('build/my-languages.so', 'python')
parser.set_language(PY_LANGUAGE)


def record_dict(candidate_dict, sol=False):
    if sol:
        solution_stmt_trace_list.append(candidate_dict)
    else:
        pred_stmt_trace_list.append(candidate_dict)


def find_blocking(code, sol=False):
    # Parse code to analyze blocks, such as 'for' and 'with' statements
    tree = parser.parse(code.encode())

    # Traverse the tree and extract blocks as code segments
    blocks = []
    for node in tree.root_node.children:
        if node.type == 'for_statement':
            block_start = node.start_byte
            block_end = node.end_byte
            block_code = code[block_start:block_end]
            blocks.append(block_code)
        elif node.type == 'with_statement':
            block_start = node.start_byte
            block_end = node.children[-1].end_byte
            block_code = code[block_start:block_end]
            blocks.append(block_code)

            # Record the block in the trace list
            record_dict(locals(), sol=sol)

    return 'record_dict(locals(), sol={})\n'.format(sol).join(blocks) + '\nrecord_dict(locals(), sol={})'.format(sol)


def trace_code(code, solution, input_globals):
    pred_stmt_trace_list.clear()
    solution_stmt_trace_list.clear()
    total_cnt = -1
    for generated_input in input_globals:
        exec_global = {'input': generated_input}
        exec_locals = {}

        pred__temp = code.replace('\\n', '\n').split('\n')
        cand_code = 'def candidate():\n' + '\n\t'.join(pred__temp) + '\ncandidate()'
        cand_code = find_blocking(cand_code)
        exec(cand_code, exec_global, exec_locals)

        sol_temp = solution.replace('\\n', '\n').split('\n')
        cand_sol = 'def candidate_sol():\n' + '\n\t'.join(sol_temp) + '\ncandidate_sol()'
        cand_sol = find_blocking(cand_sol, exec_global, exec_locals)
        exec(cand_sol, exec_global, exec_locals)

        # Find the longest common prefix of recorded states
        cnt = -1
        for idx, (pred_row, sol_row) in enumerate(zip(pred_stmt_trace_list, solution_stmt_trace_list)):
            if set(pred_row.items()) == set(sol_row.items()):
                cnt += 1
            else:
                break
        total_cnt = max(total_cnt, cnt)

    # Extract the similar prefix from the generated code
    code_idx = cand_code.find(find_blocking("", sol=False))
    return total_cnt, cand_code[:code_idx].replace('\n\t', '\n').replace('\n', '\\n')


def extract_prefix(pred_jsonl_file, sol_jsonl_file):
    prefix_list = []
    with jsonlines.open(sol_jsonl_file) as pred_reader:
        for pred_item in pred_reader:
            epoch = pred_item['epoch']
            task_id = pred_item['task_id']
            prompt = pred_item['prompt']
            completion = pred_item['completion']

            generated_code = completion
            pred_stmt_trace_list.clear()
            generated_input = pred_item['input']
            
            with jsonlines.open(pred_jsonl_file) as sol_reader:
                common_prefix_cnt_total = -1
                common_prefix_total = ""
                for sol_item in sol_reader:
                    generated_solution_code = sol_item['solution']
                    solution_stmt_trace_list.clear()

                    common_prefix_cnt, common_prefix = trace_code(generated_code, generated_solution_code, generated_input)                    
                    if common_prefix_cnt > common_prefix_cnt_total:
                        common_prefix_cnt_total = common_prefix_cnt
                        common_prefix_total = common_prefix
            prefix = {
                'epoch': epoch,
                'task_id': task_id,
                'prompt': prompt,
                'prefix': common_prefix_total.replace("def candidate():\\n", "").replace("\\ncandidate()", "").strip()
            }
            prefix_list.append(prefix)

    return prefix_list


def save_prefix_as_jsonl(prefix_list, output_file):
    with jsonlines.open(output_file, mode='w') as writer:
        for prefix in prefix_list:
            writer.write(prefix)