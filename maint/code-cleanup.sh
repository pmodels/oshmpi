#! /usr/bin/env bash

if hash gindent 2>/dev/null; then
    indent=gindent
else
    indent=indent
fi

indent_code()
{
    file=$1
    tmpfile=/tmp/${USER}.${BASHPID}.__tmp__

    $indent \
        `# Expansion of Kernighan & Ritchie style` \
        --no-blank-lines-after-declarations \
        `# --no-blank-lines-after-procedures` `# Overwritten below` \
        `# --break-before-boolean-operator` `# Overwritten below` \
        --no-blank-lines-after-commas \
        --braces-on-if-line \
        --braces-on-struct-decl-line \
        `# --comment-indentation33` `# Overwritten below` \
        --declaration-comment-column33 \
        --no-comment-delimiters-on-blank-lines \
        --cuddle-else \
        --continuation-indentation4 \
        `# --case-indentation0` `# Overwritten below` \
        `# --else-endif-column33` `# Overwritten below` \
        --space-after-cast \
        --line-comments-indentation0 \
        --declaration-indentation1 \
        --dont-format-first-column-comments \
        --dont-format-comments \
        --honour-newlines \
        --indent-level4 \
        --parameter-indentation0 \
        `# --line-length75` `# Overwritten below` \
        --continue-at-parentheses \
        --no-space-after-function-call-names \
        --no-space-after-parentheses \
        --dont-break-procedure-type \
        --space-after-for \
        --space-after-if \
        --space-after-while \
        `# --dont-star-comments` `# Overwritten below` \
        --leave-optional-blank-lines \
        --dont-space-special-semicolon \
        `# End of K&R expansion` \
        --line-length100 \
        --else-endif-column1 \
        --start-left-side-of-comments \
        --break-after-boolean-operator \
        --comment-indentation1 \
        --no-tabs \
        --blank-lines-after-procedures \
        --leave-optional-blank-lines \
        --braces-after-func-def-line \
        --brace-indent0 \
        --cuddle-do-while \
        --no-space-after-function-call-names \
        --case-indentation4 \
        ${file}

    rm -f ${file}~
    cp ${file} ${tmpfile} && \
    cat ${file} | sed -e 's/ *$//g' -e 's/( */(/g' -e 's/ *)/)/g' \
    -e 's/if(/if (/g' -e 's/while(/while (/g' -e 's/do{/do {/g' -e 's/}while/} while/g' > \
    ${tmpfile} && mv ${tmpfile} ${file}
}

usage()
{
    echo "Usage: $1 [filename | directory] {--debug}"
}

# Check usage
if [ -z "$1" ]; then
    usage $0
    exit
fi

# Make sure the parameters make sense
recursive=0
inp=
debug=
for arg in $@; do
    if [ "$arg" = "--debug" ]; then
	    debug="echo"
      continue
    fi
    # specify a directory or file
    if [ -d $arg ]; then
        recursive=1
        inp=$arg
    elif [ -f $arg ]; then
        inp=$arg
    fi
done

if [ "X$inp" = "X" ]; then
    usage $0
    exit
fi

if [ "$recursive" = "1" ]; then
    pushd $inp > /dev/null
    for i in `find . \! -type d | egrep '(\.c$|\.h$|\.c\.in$|\.h\.in$|\.cpp$|\.cpp.in$)'` ; do
        ${debug} indent_code $i
        ${debug} indent_code $i
    done
    popd > /dev/null
else
    ${debug} indent_code $inp
    ${debug} indent_code $inp
fi
