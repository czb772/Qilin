# get the time
current_time=$(date "+%Y-%m-%d %H:%M:%S")
git add .
git commit -m "update at $current_time"
git push