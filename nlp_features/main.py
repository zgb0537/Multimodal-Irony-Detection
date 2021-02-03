import sys
import nlp_features

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: main.py <file name>')
        sys.exit(0)

    features = [0,1,0,0,0,1,0,0,1,0,0]  # Would extract number of words, number of colons and number of uppercase letters
    nlp = nlp_features.start()
    with open(sys.argv[1]) as inputFile:
        counter = 0
        for line in inputFile:
            counter += 1
            nlp_features.extract(nlp, features, line, 'output-' + str(counter) + '.npy')
