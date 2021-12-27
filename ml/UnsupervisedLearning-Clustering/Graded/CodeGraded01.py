Rounded Sum
Description
You're given a list of non-negative integers. Your task is to round the given numbers to the nearest multiple of 10. For instance, 15 should be rounded to 20 whereas 14 should be rounded to 10. After rounding the numbers, find their sum.

Hint: The Python pre-defined function round() rounds off to nearest even number - it round 0.25 to 0.2. You might want to write your own function to round as per your requirement.

Sample input (a list):
[2, 18, 10]

Sample output (an integer):
30

input_list = [10,20]
#input_list = [2, 18, 10]
#input_list = [2, 14, 10]
#input_list = [2,14,15,16]
def myRound(n2r):
    if n2r%10 == 0:
        return n2r
    if n2r%5 == 0 and n2r%10 != 0:
        # return roundup
        return (n2r-(n2r%10))+10
    roundDown = (n2r // 10 )*10
    roundUp = roundDown + 10
    return (roundUp if n2r - roundDown > roundUp - n2r else roundDown)

sum=0
for num in input_list:
    n2r =  myRound(num)
    #print(n2r)
    sum = sum + n2r
result = sum
print(result)

20//10
15%10
20%10



Alarm Clock
Description
You're trying to automate your alarm clock by writing a function for it. You're given a day of the week encoded as 1=Mon, 2=Tue, ... 6=Sat, 7=Sun, and a boolean value (a boolean object is either True or False. Google "booleans python" to get a better understanding) indicating if you're are on vacation. Based on the day and whether you're on vacation, write a function that returns a time in form of a string indicating when the alarm clock should ring.

When not on a vacation, on weekdays, the alarm should ring at "7:00" and on the weekends (Saturday and Sunday) it should ring at "10:00".

While on a vacation, it should ring at "10:00" on weekdays. On vacation, it should not ring on weekends, that is, it should return "off".

----------------------------------------------------------------------
Sample input (a list):
[7,True]

Sample output (a string):
off
----------------------------------------------------------------------
Sample input (a list):
[3,True]

Sample output (a string):
10:00
----------------------------------------------------------------------
import ast,sys
input_str = sys.stdin.read()
input_list = ast.literal_eval(input_str)

#input_list=[7,True]
input_list=[3,True]
day_of_the_week = input_list[0]
is_on_vacation = input_list[1]

# write your code here
def alarm_clock(day, vacation):
    weekdays=[1,2,3,4,5]
    weekends=[6,7]
    if day in weekdays:
        if(vacation):
            return "10:00"
        else:
            return "7:00"
    else:
        if(vacation):
            return "off"
        else:
            return "10:00"

# do not change the following code
time = alarm_clock(day_of_the_week, is_on_vacation)
print(time.lower())











Sum and Squares
Description
You're given a natural number 'n'. First, calculate the sum of squares of all the natural numbers up to 'n'. Then calculate the square of the sum of all natural numbers up to 'n'. Return the absolute difference of these two quantities.

For instance, if n=3, then natural numbers up to 3 are: 1, 2 and 3. The sum of squares up to 3 will be 1^2 + 2^2 + 3^2 = 14. The square of the sum of natural numbers up to 3 is (1+2+3)^2=36. The result, which is their absolute difference is 22.

Sample input (an integer):
3

Sample output (an integer):
22

n=3
sumOfSq = 0
sqOfSum = 0
for i in range(1,n+1):
    sumOfSq = sumOfSq + i**2

onlySum=0
for j in range(1,n+1):
    onlySum+=j
sqOfSum = onlySum**2
abs_difference = (sqOfSum-sumOfSq)
print(abs_difference)
