. venv/bin/activate

pip freeze > requirements.txt

git commit -m 'deployment dependencies'

git push

git checkout main

git merge dev

git push

