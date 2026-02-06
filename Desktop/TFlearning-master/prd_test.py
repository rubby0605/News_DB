import pandas_datareader as pdr
from datetime import datetime
start = datetime(2020,6,6)
end = datetime(2020,6,10)
goog=pdr.data.DataReader("MSFT",'google',start,end)
goog[:5]