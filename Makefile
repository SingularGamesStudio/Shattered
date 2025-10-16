.PHONY:add
add:
	cmd //C tree Assets/Scripts //f > tree.txt
	git add *
	git add -u