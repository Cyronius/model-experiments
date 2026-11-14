"""Deterministic arithmetic oracles and hand-specified algorithm curricula."""
from fractions import Fraction
import random
import math
from broad_healing import program

FAMILIES=['add_subtract','multiply','divide','fraction','percent','equation','remainder','multi_step']

def math_example(rng, family):
    a,b,c=rng.randint(12,98),rng.randint(3,29),rng.randint(2,19)
    if family=='add_subtract':
        values=[a,b,c];expr=f'{a}+{b}-{c}';ans=a+b-c
        prompt=f'Compute ({a} + {b}) - {c}.'
        work=f'First add: {a} + {b} = {a+b}. Then subtract {c}: {a+b} - {c} = {ans}.'
        transfer=f'A box contains {a} counters. Add {b} counters and then remove {c}. How many remain?'
    elif family=='multiply':
        values=[a,b];expr=f'{a}*{b}';ans=a*b;t,u=divmod(b,10)
        prompt=f'Calculate {a} times {b}.'
        work=f'{b} = {10*t} + {u}. The partial products are {a} × {10*t} = {a*t*10} and {a} × {u} = {a*u}. Their sum is {ans}.'
        transfer=f'There are {b} trays with {a} beads in each tray. How many beads are there altogether?'
    elif family=='divide':
        values=[a*b,b];expr=f'{a*b}/{b}';ans=a
        prompt=f'Divide {a*b} by {b}.'
        work=f'{b} × {a} = {a*b}, so {a*b} divided by {b} is {a} with no remainder.'
        transfer=f'Share {a*b} tokens equally among {b} boxes. How many tokens go in each box?'
    elif family=='fraction':
        d=rng.randint(2,13);a=rng.randint(1,12);values=[a,b,c,d];expr=f'{a}/{b}+{c}/{d}';ans=Fraction(a,b)+Fraction(c,d)
        prompt=f'Add the fractions {a}/{b} and {c}/{d}.'
        work=f'A common denominator is {b*d}. The numerator is {a} × {d} + {c} × {b} = {a*d+c*b}. Reducing {a*d+c*b}/{b*d} gives {ans}.'
        transfer=f'A ribbon is {a}/{b} meter long and another is {c}/{d} meter long. What is their combined length in meters?'
    elif family=='percent':
        p=rng.choice([5,10,15,20,25,40,50,75]);v=rng.randint(2,30)*20;values=[p,v];expr=f'{p}*{v}/100';ans=Fraction(p*v,100)
        prompt=f'Find {p}% of {v}.'
        work=f'{p}% means {p}/100. Multiplying {v} by {p}/100 gives {ans}.'
        transfer=f'A collection has {v} cards, and {p}% are blue. How many cards are blue?'
    elif family=='equation':
        x=rng.randint(-20,40);v=b*x+c;values=[b,c,v];expr=f'({v}-{c})/{b}';ans=x
        prompt=f'Solve {b}*x + {c} = {v} for x.'
        work=f'Subtract {c} from both sides: {b}*x = {v-c}. Divide by {b}: x = {x}.'
        transfer=f'A number multiplied by {b}, then increased by {c}, becomes {v}. What is the number?'
    elif family=='remainder':
        values=[a,b];expr=f'{a}-{a//b}*{b}';ans=a%b
        prompt=f'What is the remainder when {a} is divided by {b}?'
        work=f'{a} = {a//b} × {b} + {ans}. The remainder is {ans}, which is nonnegative and less than {b}.'
        transfer=f'Pack {a} marbles into full bags of {b}. How many marbles are left unpacked?'
    else:
        values=[a,b,c];expr=f'{a}*{b}+{c}';ans=a*b+c
        prompt=f'Calculate {a}*{b} + {c}.'
        work=f'Multiply first: {a} × {b} = {a*b}. Add {c}: {a*b} + {c} = {ans}.'
        transfer=f'A club buys {b} packs containing {a} stickers each and receives {c} extra stickers. How many stickers does it have?'
    return {'family':family,'values':values,'expression':expr,'expected':str(ans),'prompt':prompt,'work':work,'transfer':transfer}


def training_algorithms():
    # Fixed targets with independently generated multi-input checks. These do
    # not implement the old or new held-out functions.
    rng=random.Random(261026)
    arrays=[[],[0],[-2,0,3],[4,4,1]]+[rng.choices(range(-9,10),k=rng.randrange(1,12)) for _ in range(12)]
    specs=[
      ('prefix_sum','main_solution(values)','Return the inclusive running sums of the integer list values.',
       'def main_solution(values):\n    result = []\n    total = 0\n    for value in values:\n        total += value\n        result.append(total)\n    return result',
       [({'values':a},[sum(a[:i+1]) for i in range(len(a))]) for a in arrays]),
      ('even_filter','main_solution(values)','Return just the even integers from values in their original order.',
       'def main_solution(values):\n    return [value for value in values if value % 2 == 0]',
       [({'values':a},[x for x in a if x//2*2==x]) for a in arrays]),
      ('clamp','main_solution(values, low, high)','Given low <= high, replace values below low by low and values above high by high; preserve other values and order.',
       'def main_solution(values, low, high):\n    result = []\n    for x in values:\n        if x < low:\n            result.append(low)\n        elif x > high:\n            result.append(high)\n        else:\n            result.append(x)\n    return result',
       [({'values':a,'low':-3,'high':5},[min(5,max(-3,x)) for x in a]) for a in arrays]),
      ('gcd','main_solution(a, b)','Return the greatest common divisor of nonnegative integers a and b; return 0 if both are zero.',
       'def main_solution(a, b):\n    while b:\n        a, b = b, a % b\n    return a',
       [({'a':a,'b':b},math.gcd(a,b)) for a,b in [(0,0),(0,9),(12,18),(17,23)]+[(rng.randrange(100),rng.randrange(100)) for _ in range(12)]]),
      ('prime','main_solution(n)','Return whether integer n is prime. Integers below 2 are not prime.',
       'def main_solution(n):\n    if n < 2:\n        return False\n    d = 2\n    while d * d <= n:\n        if n % d == 0:\n            return False\n        d += 1\n    return True',
       [({'n':n},n>=2 and all(n%d for d in range(2,n))) for n in [-4,0,1,2,3,4,25,49,97,101,121]+rng.sample(range(5,150),12)]),
      ('digit_sum','main_solution(n)','Return the sum of decimal digits of integer n, ignoring its sign.',
       'def main_solution(n):\n    n = abs(n)\n    total = 0\n    while n:\n        total += n % 10\n        n //= 10\n    return total',
       [({'n':n},sum(map(int,str(abs(n))))) for n in [0,-123,9001]+[rng.randint(-99999,99999) for _ in range(12)]]),
      ('reverse_words','main_solution(text)','Split text on whitespace, reverse the order of words, and join with one space. Return an empty string if there are no words.',
       'def main_solution(text):\n    words = text.split()\n    words.reverse()\n    return " ".join(words)',
       [({'text':s},' '.join(s.split()[::-1])) for s in ['', '  ', 'a','red blue green',' a\t b\n c ', 'one two one']]),
      ('positive_total','main_solution(values)','Return the sum of strictly positive values in the integer list values. Return zero if none are positive.',
       'def main_solution(values):\n    total = 0\n    for value in values:\n        if value > 0:\n            total += value\n    return total',
       [({'values':a},sum(max(x,0) for x in a)) for a in arrays]),
      ('zip_add','main_solution(left, right)','Given equally long integer lists left and right, return their elementwise sums.',
       'def main_solution(left, right):\n    return [a + b for a, b in zip(left, right)]',
       [({'left':a,'right':a[::-1]},[a[i]+a[-i-1] for i in range(len(a))]) for a in arrays]),
      ('adjacent_difference','main_solution(values)','Return each element minus its predecessor, starting at the second element. Lists shorter than two elements return [].',
       'def main_solution(values):\n    return [values[i] - values[i - 1] for i in range(1, len(values))]',
       [({'values':a},[b-a for a,b in zip(a,a[1:])]) for a in arrays]),
      ('count_above','main_solution(values, threshold)','Count the entries of values strictly greater than threshold.',
       'def main_solution(values, threshold):\n    count = 0\n    for value in values:\n        if value > threshold:\n            count += 1\n    return count',
       [({'values':a,'threshold':2},len([x for x in a if x>2])) for a in arrays]),
      ('square_map','main_solution(values)','Return the square of every integer in values, preserving order.',
       'def main_solution(values):\n    return [value * value for value in values]',
       [({'values':a},[x**2 for x in a]) for a in arrays]),
    ]
    return specs


def code_eval():
    def p(id,signature,task,tests):
        return program(id,f'Use exactly the function signature {signature}. {task}',tests)
    dev=[p('cdev-product','main_solution(values)','Return inclusive cumulative products. An empty list returns [].',[[{'values':[2,3,4]},[2,6,24]],[{'values':[]},[]],[{'values':[2,0,5]},[2,0,0]],[{'values':[-2,3,-1]},[-2,-6,6]]]),
         p('cdev-palindrome','main_solution(text)','Return whether text reads identically forwards and backwards. Matching is case-sensitive and includes all characters.',[[{'text':'abba'},True],[{'text':'abca'},False],[{'text':''},True],[{'text':'Aa'},False]])]
    test=[p('ctest-dot','main_solution(left, right)','Given equal-length numeric lists, return their dot product. Empty lists return 0.',[[{'left':[1,2,3],'right':[4,5,6]},32],[{'left':[],'right':[]},0],[{'left':[-1,2],'right':[3,-4]},-11]]),
          p('ctest-unique','main_solution(text)','Return the first character occurring exactly once in text. Return None if there is no such character. Matching is case-sensitive.',[[{'text':'swiss'},'w'],[{'text':'aabb'},None],[{'text':''},None],[{'text':'AaA'},'a']]),
          p('ctest-streak','main_solution(values)','Return the length of the longest consecutive run of strictly positive numbers. Zero and negative numbers break a run.',[[{'values':[1,2,0,3,4,5,-1]},3],[{'values':[]},0],[{'values':[-1,0]},0],[{'values':[2,3]},2]]),
          p('ctest-overlap','main_solution(text, pattern)','Count occurrences of nonempty pattern in text, including overlapping occurrences.',[[{'text':'aaaa','pattern':'aa'},3],[{'text':'abababa','pattern':'aba'},3],[{'text':'abc','pattern':'z'},0],[{'text':'','pattern':'a'},0]])]
    return dev,test
