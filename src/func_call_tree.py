import ast

tree = ast.parse(open('run_isa.py').read())
print(ast.dump(tree))  # dumps the whole tree

# get the function from the tree body (i.e. from the file's content)
func = tree.body

# print(func)

# get the function argument names
for a in func:
    arguments = []
    try:
        arguments.append(a.arg.arg.arg)
    except Exception:
        pass

print('the functions is: %s(%s)' % (func.name, ', '.join(arguments)))