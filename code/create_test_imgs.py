from IPython.lib.latextools import latex_to_png
import matplotlib.pyplot as plt
import numpy as np
import os


def create_latex(filename, eq, fontsize=50, figsize=(5,5)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.text(0.5, 0.5, '$%s$' % eq, fontsize=fontsize,
            ha='center', va='center')
    ax.axis('off')
    plt.savefig(filename)
    plt.close(fig)


def create_equation(symbols, depth=0, p=np.array([0., 1.]), eq=True):
    choice = np.random.choice(range(2), p=p)

    if choice == 0:     # terminate on symbol
        max_size = 3
        numbers = ''.join([symbols[0][np.random.randint(len(symbols[0]))]
                           for _ in xrange(np.random.randint(max_size+1))])
        if len(numbers) < max_size:
            max_size = max_size - len(numbers)
            letters = ' '.join([symbols[1][np.random.randint(len(symbols[1]))]
                               for _ in xrange(np.random.randint(1, max_size+1))])
        else:
            letters = ''
        
        return numbers + letters, depth

    operators = ['{{{left}}}_{{{right}}}', '{{{left}}}^{{{right}}}', 
                 '{left}+{right}', '{left}-{right}', '{left}*{right}', 
                 '\\frac{{{left}}}{{{right}}}']
    equalities = ['{left}={right}', '{left}>{right}', '{left}<{right}', 
                  '{left}\leq {right}', '{left}\geq {right}']
    
    if eq:
        operators.extend(equalities)

    # adjust probability
    adjust = 0.5
    p = p + (np.array([1,-1]) * adjust)

    # select operator
    op = np.random.choice(operators)

    # keep equality operators?
    eq = eq & (op in ['{}+{}', '{}-{}', '{}*{}'])
    
    equation_l, depth_l = create_equation(symbols, depth+1, p, eq)
    equation_r, depth_r = create_equation(symbols, depth+1, p, eq)

    return op.format(left=equation_l, right=equation_r), max(depth_l, depth_r)
    

def main():
    paths = ['imgs/numbers/',
             'imgs/letters/lower/',
             'imgs/letters/upper/',
             'imgs/letters/greek_lower/',
             'imgs/letters/greek_upper/']

    numbers = os.listdir(paths[0])
    letters = []
    for path in paths[1:]:
        letters.extend(os.listdir(path))

    numbers = [n.split('_')[0] for n in numbers]
    letters = [l.split('_')[0] for l in letters]

    symbols = [numbers] + [letters]
    
    num_tests = 10000
    equations = []
    depths = []
    filenames = []
    for i in xrange(num_tests):
        eq, depth = create_equation(symbols)
        num = str(i)
        num = ('0' * (4 - len(num))) + num
        filename = 'test_imgs/test_{}.png'.format(num)
        print i, '------', eq
        create_latex(filename, eq, fontsize=100, figsize=(15, 7))
        equations.append(eq)
        depths.append(depth)
        filenames.append(filename)

    with open('test_imgs.csv', 'w') as f:
        f.write('equation,depth,filename\n')
        for i in xrange(num_tests):
            line = '"{}",{},"{}"\n'.format(equations[i], 
                                           depths[i], 
                                           filenames[i])
            f.write(line)


if __name__ == '__main__':
    main()
