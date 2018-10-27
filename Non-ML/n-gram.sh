cat ita.txt | python process.py | ./kenlm/build/bin/lmplz -o 3 > ita.arpa
cat deu.txt | python process.py | ./kenlm/build/bin/lmplz -o 3 > deu.arpa
cat eng.txt | python process.py | ./kenlm/build/bin/lmplz -o 3 > eng.arpa
cat fra.txt | python process.py | ./kenlm/build/bin/lmplz -o 3 > fra.arpa
cat por.txt | python process.py | ./kenlm/build/bin/lmplz -o 3 > por.arpa
cat spa.txt | python process.py | ./kenlm/build/bin/lmplz -o 3 > spa.arpa



