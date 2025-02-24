# ---

# # Cheat sheet of essential Bash/Linux shell commands
#
# If you're new to shell commands, here's a list of useful commands to help you get started. Note that it's best to execute these commands in a terminal.
#
# ---

# **Change directory**

cd ~/

# The left part of the path will change the behavior of the above command
#
# * ```~``` refers to your home directory
# * ```/```  refers to the root (top-level) directory
# * ```.``` refers to the working (current) directory
# * ```..``` refers to the parent (of the working) directory

# **Print working directory**

pwd

# **List files and child directories in working directory**

ls

# **Create a new directory**
#
# note: -p creates intermediate folders if they do not exists

mkdir -p myfolder

# create a file with some content
echo -e "HELLO\nWORLD" > myfolder/myfile

ls ./myfolder

# **Copy file**

cp ./myfolder/myfile ./myfolder/copyfile

ls ./myfolder

# **Move file**
#
# Copy and move will replace the destination file if it exists, unless you specify -i on the command line, e.g. `mv -i fromfile tofile`

mv ./myfolder/copyfile ./myfolder/myfile

ls ./myfolder

# **View content of file**

cat ./myfolder/myfile

# Note: other commands
#
# * `less`: View file content (with scroll).
# * `head`: View first 10 lines
# * `tail`: View last 10 lines

# **Delete file (careful)**

rm ./myfolder/myfile

# **Delete empty folder**

rmdir ./myfolder

# **Get help**
#
# Most commands will display a short help page if invoked with the `--help` (or -h) option. Additionally, the `man <command>` command will display the full details.

ls --help

man ls


