

def time_delta_str(seconds) -> str:
  h = int(seconds/(60*60))
  m = int( (seconds - (h * 60 * 60)) / 60 )
  s = round( seconds - (m * 60), 2 )
  delta = "{} hours(s) {} minute(s) {} second(s)".format(h, m, s)
  return delta