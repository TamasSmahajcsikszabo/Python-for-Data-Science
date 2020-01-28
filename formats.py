#! python

def commas(N):
    digits = str(N)
    assert digits.isdigit()
    result = ''
    while digits:
        digits, last3 = digits[:-3], digits[-3:]
        result = (last3 + ',' + result`) if result else last3
    return result

commas(78676134433)
