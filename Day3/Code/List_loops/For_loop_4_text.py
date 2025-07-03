
''' py
Homework from Day 04 due 2019.04.11
'''
disney_characters = ["simba", "ariel", "pumba", "flounder", "nala", "ursula", "scar", "flotsam", "timon"]

for char in disney_characters:
	if 'u' in char:
		print(char + " U are so uniquely U!")
		continue
	elif  'i' in char:
		print(char + " I bet you're Impressively Intelligent!")
		continue
	elif 'o' in char:
		print(char + "O My! How Original!")
		continue
	else:
		if 'a' in char or 'e' in char:
			print(char + " Ehh, a's and e's are so ordinary.")
		else:
			continue