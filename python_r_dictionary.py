#R to python dictionary
#frank hm wong
#12 sep 2018


#If you come from inferno and want to learn some python, 
#you need a dictionary too.                                

#this is the collection of my SO search results for 'python's equivalent of R's ...'
#Please send comments to f.wong at ymail.com, SO links are very appreciated.

#Also check Pandas-R cheat sheet by Giuseppe Paleologo
#https://sites.google.com/site/gappy3000/home/pandas_r


#Notations
#Python
#d/df pd.DataFrame 
#s pd.Series
#arr numpy.array
#v numpy 1d array


#R:
#df: data.frame 
#d/DT: data.table
#x : xts
#v : vector


#R: getwd(); setwd(path_wd)
os.getcwd()
os.chdir(path_wd)

#R: nrow(X);ncol(X);dim(X)     #length(X) is number of col
len(df); shape(df)[1]; shape(df)  #len(X) is number of row

#select col
d.Month     #R: d$Month
d['Month']   #R:d[['Month']]
d[['Month','Year']] #R:df[c("Month","Year")] #or d[, c("Month","Year")]


#select row
d[:3]   #R: d[1:3,]



d.index     #R: rownames(d) #index(x)
d.columns    #R: colnames(d)
d.index.tolist()


#select by rownames/colnames

#R: df[1:4, c('a','b')] #R accept string rowname/colnames or boolean or integer position 
#.loc accept index or boolean
#.iloc accept position 

#select by rownames (index for pandas)
#df[c('1','2','3'),] #R df rownames are always string
d.loc[0:3]      #find rows where index in 0:3 , not first 4 rows
d.loc['20130201':'20130401'] #if index is date
d.loc[['a','b','c']]    #match a,b,c to index

#R:df[1:4,]
d.iloc[0:3,:]     #first 4 rows

#R:df[c(TRUE,FALSE),]
d.loc[[True, False],:]

#R: d[1:4, c('a','b')]  #1:4 are row position
d[['a','b']].iloc[0:3,:]
d.loc[:,['a','b']].iloc[0:3,:]

#select by booean 
d.loc[[True,False,True]]        
d.iloc[[True,False,True]]

#R: d[a==5,] / df[df$a ==5,]
d.loc[d['a']==5,]


#select col
#R: d[c("a","b","c")], 
#select col : R: df['a', drop= FALSE], d[['a']]
d.loc[:, ['c','d']]
d.loc[:, 'c'] #drops to Series
d.loc[['c']] #stays DF


#convert to numeric
#R: as.numeric(v)
pd['a'].astype(float)
pd['a'].astype(int)
pd.to_numeric(v)



#multiply by col (R store by col, np store by row)
#https://stackoverflow.com/questions/18522216/multiplying-across-in-a-numpy-array
#R: d * v
df.mul(v, axis=0)
arr * v[:,newaxis]

#multiply by row
#R: data.frame(mapply('*', d, v, SIMPLIFY=FALSE))
df.mul(v, axis=1)
arr * v



#remove rows by boolean
#R: d[a>5]
#https://stackoverflow.com/questions/13851535/how-to-delete-rows-from-a-pandas-dataframe-based-on-a-conditional-expression
d2 = d.drop(d[d["a"]>5].index)
d.drop(d[d["a"]>5].index,inplace=True)

df = df.drop(df[(df.score < 50) & (df.score > 20)].index)

#R:d[c(True,False,True),]

#and or logical 
#https://stackoverflow.com/questions/33384529/difference-between-numpy-logical-and-and
#R: a & b
(arr1) & (arr2) #brackets are important if anything in inbetween, works on boolean only
np.logical_and(arr1,arr2) #work on everything

#R: seq(1, 100, by = 0.5)
np.arange(1,100.5, 0.5)   #not a typo
#R: seq(1, 10, length.out = 20)
numpy.linspace(1,10, 20)  #start, stop, num 

#R: rep(1:2,10)
np.tile([1,2],10)
np.array([1,2,1,2,1,2....])
#np.repeat([1,2],3) = np.array(1,1,1,2,2,2)

#numpy.ufunc.reduce
#R: Reduce("+", list(a,b,c))
np.add.reduce([a,b,c])

#R: Reduce("|", list(a,b,c))
np.logical_or.reduce([a,b,c])

#R: pmax(a,b)
np.maximum(a,b) 

#R: max(c(1,2,3,4))
np.max([1,2,3,4])

#R: %in% 
d[d['a'].isin(['x','y'])]   #R: d[ d$a %in% c("x","y")]
np.isin(arr, [1,2])

v = np.array([1,2,3,4])
u = np.array([1,2])
np.isin(v,u)
#TTFF

#== to single value
#R: v == 7
d['a'] == 7   #return Series
s == 7        #return Series
arr == 7      #return np.array
[1,2,3,4] ==7 #return False (comparing list to scalar)


d.dtypes   #R:str(d)

#construct R named list/named vector
#constructed python dictionary 
#R: v = c(a=1,b=2) #named list
#R: list(a=1,b=2)  #named list
{'a':1, 'b':2}     #dict
dict(a=1,b=2)      #dict
from collections import OrderedDict
OrderedDict( [('a',1),('b',2)])




#R: cbind(a,b)
#https://stackoverflow.com/questions/32801806/pandas-concat-ignore-index-doesnt-work
#if index arent same
#a.reset_index(drop=True, inplace=True)
#b.reset_index(drop=True, inplace=True)
pd.concat([a,b], axis=1).reset_index() #if index are different, they merge by index
pd.concat([a.reset_index(),b.reset_index()], axis=1).reset_index() #if their index does not match 

#https://stackoverflow.com/questions/21887754/numpy-concatenate-two-arrays-vertically
np.column_stack([arr1,arr2])
np.hstack([arr1[:,None],arr2[:,None]])

np.concatenate([arr1,arr2], axis=1) #does not work

#R: rbind(a,b)
#numpy arrange by row, so, 1d vector is a row
pd.concat([a,b], axis=0).reset_index() 
np.concatenate([arr1,arr2], axis=0) #default is rbind(axis=0)
np.vstack([arr1[None,:], arr2[None,:])#stack vertically 

#cbind
np.hstack([a,b])
#rbind
np.vstack([a,b])
#R: do.call(cbind,...)
np.hstack([a,b,c])
#R: do.call(rbind,...)
#R: rbindlist(list_d)
np.vstack([a,b,c])
pd.concat(list_d,axis=0)


#cat 1d np array
#R: c(v1,v2)
np.append(arr, arr2)
np.hstack([np.arange(10),np.arange(20)])
np.r_[[1,2,3],[4,5,6]]

#more on r_ , concatenate, hstack
#https://stackoverflow.com/questions/37743843/python-why-use-numpy-r-instead-of-concatenate

#1d to 2D
#1d do n x 1
#R: cbind(v)
#R: matrix(v, ncol=1)
#R: data.frame(v)
arr[:,np.newaxis]
arr[:,None] #None is np.newaxis
arr.reshape([len(arr),1])
arr.reshape([-1,1]) 

#1d to 1 x n
arr[np.newaxis,:]
arr[None,:] #None is np.newaxis
arr.reshape([1,len(arr)])
arr.reshape([1,-1])



#merge
#R: merge(d1,d2, by = "name")
pd.merge(d1,d2, on = "name")

#R: merge(d1,d2,by.x = "name_a", by.y = "name_b")
pd.merge(d1,d2, left_on = "name_a",right_on = "name_b")

#R: Reduce(function(x,y)merge(x,y, by = "A"), list_df)
#https://stackoverflow.com/questions/38089010/merge-a-list-of-pandas-dataframes
from functools import reduce
reduce(lambda x,y: pd.merge(x,y, on = "A"), list_df)

#left join without copy
#R: d[right, on = "a", b:=i.b]
djoin = pd.merge(d[['a']], right.drop_duplicates(subset='a'), on = 'a', how = "left")  #fail when duplicate keys in right
d['b'] = djoin['b'].values

d['b'] = np.nan
d.set_index('a')
right.set_index('a')
d.update(right)
d.reset_index(level=0, inplace=True)
#update doesnt work because it updates only the first ones


#head/tail

#R: head(d)/tail(d)
d.head()
d.tail()



#functional
#DF -> vector
#R: sapply(DF, f) #f: vector -> scalar
np.apply_along_axis(f, 0, arr)
DF.apply(f)

#DF -> DF
#R: as.Data.Frame(lapply(DF, f)) #f: vector-> vector of same len
np.apply_along_axis(f, 0, arr)
DF.transform(f)
DF.apply(f)

#select ith item from named list 
#select ith item from dict
#R: lapply(l, "[[", 2)  
{k:v[1] for k,v in dict1.items()}



#rename columns
#R:  colnames(df) = c('a','b')
#R: setnames(DT, cname_old, cname_new)

df.rename(columns = {'oldname1':'newName1', 'oldName2':'newName2'}, inplace = True)
#http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.rename.html



#remove column
#R: d[['a']] <- NULL
d.drop(['a','b'], inplace = True, axis = 1)
d.drop(columns = ['a','b'], inplace = True)


#create matrix from vector
np.arange(10)               #c(0:9)
np.arange(10).reshape(5,2)  #matrix(0:9, nrow =5)

#column is datetime: lapply(d, function(v)class(v)%in% "POSIXct")
d.select_dtypes(include=['datetime64']).columns
d['DATE'] = pd.DatetimeIndex(d['index']).normalize() #also is 'datetime64'


#convert all (string)columns looks like datetime
#https://stackoverflow.com/questions/18776878/finding-columns-which-contain-dates-in-pandas
df.convert_objects(convert_dates=True)
pd.to_datetime(d['index'])                  #as.POSIXct(v)

#add one hour in time
#R: v + 60*60 # POSIXct is in number of seconds
s + pd.Timedelta(seconds = 60*60)

#R: shift(v)
s.shift(1)
#https://stackoverflow.com/questions/30399534/shift-elements-in-a-numpy-array
#for numpy array
def shift(arr, num=1, fill_value=np.nan):
    # preallocate empty array and assign slice by chrisaycock
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result = arr
    return result


#read sqlite to df
import pandas as pd
import sqlalchemy
db_name = "tx.sqlite"
table_name = "LITTLE_BOBBY_TABLES"
engine = sqlalchemy.create_engine("sqlite:///%s" % db_name, execution_options={"sqlite_raw_colnames": True})
df = pd.read_sql_table(table_name, engine)

#write df to sqlite


#regular expression
#R: grepl ("c$", c('abc','aa','abcd'))
import re
[re.search("abc$", x) is not None for x  in ['abc','aa','abcd']]
#R: grep("c$", c('abc','aa','abcd'),value = TRUE)
[x for x in ['abc','aa','abcd'] if re.search("abc$", x)]

#R: unique(v)
list(set(v)) #for list and np array
s.unique()   #for series
np.unique(arr)

#sort(v)
np.sort(v)
u = sorted(v)
print(u)
v.sort()
print(v)



a = [1,2,3]
b = [4,5,6]
#rbind(a,b)
np.array([a,b])
pd.DataFrame([a,b])

#cbind(a,b)
np.array(list(zip(a,b)))    #list() for python3 
pd.DataFrame(list(zip(a,b)))

#list to data frame
#R: data.frame(l) / data.table(l)
pd.DataFrame(dict1)
pd.DataFrame(dict1, index=[0]) #if dict is dict of scalar

#mod numpy
#R: v %% 5
np.mod(arr, 5) #np.remainder() is the same
s.mod(5)

#get ith element
#R: v[0], v[length(v)]
s[0]
s.iloc[0]
s.iloc[-1]

#named vector vs dict/dictionary
#R: names(a) <- name_a
#R: c(a=1,b=2,c=3)
dict(zip(name_a, a))
dict(a=1,b=2,c=3)
{"a":1, "b":2,"c":3}


#by , groupby
#R: d[, .(OPEN=OPEN[1], HIGH= max(HIGH), VOLUME= sum(VOLUME)), by = DATE]
dday = d.groupby(['DATE']).agg({'OPEN':'first','HIGH':'max','VOLUME':'sum'}).reset_index()

#https://pandas.pydata.org/pandas-docs/stable/groupby.html
#Note that groupby will preserve the order in which observations are sorted within each group. 
#which is same as behaviour of data.table if groupby(..., sort=False) is used

#groupby apply custom func
#R: q = quote(OPEN=OPEN[1], HIGH= max(HIGH), VOLUME= sum(VOLUME))
#R: d[, eval(q),by = DATE]

def ftmp(g):
    s = pd.Series( dict(
            OPEN = g.OPEN.iloc[0],
            HIGH = g.HIGH.max(),
            VOLUME = g.VOLUME.sum(),
        ))
    return(s)

dday = d.groupby(["DATE"],sort=False).apply(ftmp).reset_index()



#Null , NA
#as a general rule, never do logical operation on it
#only one type of it, all int convert to float
#always use pd.isnull to check it , never use np.isnan (fails on string )
[np.nan ==False,np.nan==True, np.nan==np.nan, np.nan==None, pd.isnull(np.nan)]
#have to use np.logical_and()

#remove NA
#R: v[!is.na(v)]
arr[~np.isnan(arr)]
s[!pd.isnull(s)]

#R: d[complete.cases(d)]
d[~pd.isnull(d).any(axis=1)]

#datetime
#as.Date()
#R: as.Date(v)
d['Time'].dt.normalize() #datetime
d['Time'].dt.date        #date
pd.DatetimeIndex(v).normalize()

#R: as.POSIXct(x, format = "%Y-%d-%m", tz = "GMT")
pd.to_datetime(x)
pd.to_datetime(x, format = "%Y-%d-%m")
pd.to_datetime('20140101')

#datetime from sec from epoch
#R: as.POSIXct(x, tz = "GMT", origin = ISOdate(1970,1,1,0,0,0))
pd.to_datetime(1490195805, unit='s')

#Date from year, month, day ...
#R: ISOdate(1990, c(11,12), c(1,2),0,0,0)  #default time is 12:00 GMT
pd.to_datetime(d) # d contains columns year, month, day, hour ...

#datetime to string 
#R: strftime(x, "%Y-%d-%m")
s.dt.strftime("%Y-%d-%m")

#date seq
#seq(as.Date("2016-01-01"),length.out = 10, by = 1)
pd.to_datetime(range(10), origin="2016-04-30",unit="D")
pd.date_range('2014-01-01 05:00:00', '2014-01-01 05:10:00',freq='1MIN')

#time zone, timezone
#R: attributes(v)$tzone <- "GMT"  #v is a POSIXlt vector
d['Time2'] = d['Time'].dt.tz_localize("GMT")

#collapse string
#R: paste0(v, collapse  = "_")
s.str.cat(sep="_")

#paste two string
#R: paste(a,b, sep = " ")
s.str.cat(s2, sep = " ") #return a series
np.char.add(arr1,arr2)
d['a'] + '_' + d['b'] 

#stlit string
#R: tstrsplit("_")[1:3]
s.str.split("_", expand = True, n=2) #gives a DF

#R: substr(v,1,2)
s.str.slice(0,2)

#R: summary(v)  #quantile
s.describe(include="all")
DF.describe(include="all")

#R: quantile(v, c(0.5,0.75,0.9))
np.percentile(v, [50,75,90]) #actually percentile  50 
s.quantile([0.5,.75,0.9]),   #fraction             0.5
#R: table(v) 
s.value_counts()
np.unique(arr, return_counts=True)
pd.value_counts(arr)

#R:which.max(v) / which.min(v)
s.idxmax()

#R:max(v)
s.max()
np.argmax(v)

#R: which(c(1:10)>5)
#https://stackoverflow.com/questions/33747908/output-of-numpy-wherecondition-is-not-an-array-but-a-tuple-of-arrays-why
np.where(np.arange(1,11) > 5)[0] # where returns a tuple

#R: ifelse(v1>2, v2,v3)
np.where(arr1>2, arr2,arr3)
np.where(1>2,2,3) #gives array(3) convert scalar to array

#R: v = na.locf(u)
s.ffill(inplace=True)
#R: d[, v:= na.locf(u), by = gp_a]  #groupby then na.locf
df['u'] = df.groupby('gp_a')['u'].ffill()

#sample 1:10 (0 to 9 for pandas)
#R: sample(10, 5, replace = FALSE)    #default replace = False 
np.random.choice(5,3, replace = False)  #default replace = True

#R: sample(v, 5)
np.random.choice(v, 5, replace = False)

#R: sample(10)
np.random.permutation(10) 
x = np.arange(10)
np.random.shuffle(x)  #random in place
np.random.permutation(x) #return copy of array

#R:rnorm(100)
np.random.normal(size=100)

#shuffle 2d array
#R: d[sample(1:dim(d)[1]),]
#https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
d.sample(frac=1)
from sklearn.utils import shuffle
d = shuffle(d)
d.iloc[np.random.permutation(len(d))]
arr[np.random.permutation(arr.shape[0]),:]

#https://stackoverflow.com/questions/5954603/transposing-a-numpy-array
#R:transpose(m)
arr.T
arr[np.newaxis]
arr.reshape([10,1]) #reshape of 2D objects is different from transpose

#count by group
#R:d[, a:=1:.N, by = g]
d['a']= d.groupy('g').cumcount()

#R: d[, b:=.GRP, by = g]
#https://stackoverflow.com/questions/41594703/pandas-assign-an-index-to-each-group-identified-by-groupby
df.groupby(['a']).ngroup()
df.groupby(['a', 'c']).ngroup()

#https://stackoverflow.com/questions/36063251/python-pandas-how-can-i-group-by-and-assign-an-id-to-all-the-items-in-a-group
df["b"] = LabelEncoder().fit_transform(df['g'])     #int count from 0
#https://stackoverflow.com/questions/41594703/pandas-assign-an-index-to-each-group-identified-by-groupby
df['b'] = pd.Categorical(df['a'].astype(str)).codes
df['b'] = pd.Categorical(df['a'].astype(str) + df['c'].astype(str)).codes #allow multiple col groups



#R: ind = order(v)
y = np.argsort(v)
y = v.argsort()

#R: match(v1, vdict) -> vdict[match(v1,vdict)] gives v1
np.searchsorted(vdict,v1) #if vdict is sorted
vdict[np.searchsorted(vdict,v1)] #gives v1
pd.match([1,2,3,5,8,2],[1,2,4,5,9,2])
match(c(1,2,3,5,8,2),c(1,2,4,5,9,2))

#R: d[order(v),]
d.reindex(np.argsort(d['c'])).reset_index(drop=True)
#R: setcolorder(d,new_col_order)
#https://stackoverflow.com/questions/13148429/how-to-change-the-order-of-dataframe-columns
#d.reindex_axis(['a','b','c'], axis=1) #deprecated
d.reindex(['a','b','c'], axis=1)  #copy all data

d.sort_values(["a","b"], ascending = [True,False], inplace=False)
d.sort_values("a", ascending = True) #inplace is False default

s.sort_values() #series no need to add input
d['b'] = d.a.sort_values() # error merge to d by index, undoing the sort
d['b'] = d.a.sort_values().values  #correct
#the index is also sorted #add d.reset_index(drop=True, inpalce=True) 

#R: duplicated(x)
#https://stackoverflow.com/questions/11528078/determining-duplicate-values-in-an-array
from collections import Counter
[item for item, count in Counter(a).iteritems() if count > 1]

def get_dup_items(a):
    #https://stackoverflow.com/questions/11528078/determining-duplicate-values-in-an-array
    from collections import Counter
    return([item for item, count in Counter(a).items() if count > 1])
    

#add column by group
#R: d[, a:= mean(x), by = b]
d['a'] = d.groupby("b")['x'].transform(mean)
d['a'] = d.groupby("b")['x'].transform(lambda x: x*2+1)

#R: d[, a:= 1:.N, by = b]
d["a"] = d.groupby("b")["x"].transform(lambda x: np.arange(len(x)))

#R: xts::to.min(x)
d.resample("1MIN").ohlc()
d["Time"]  = d["Time"].dt.round("1MIN")

#R: identical(x,y)
np.array_equal([1, 2], [1, 2])  #cannot check np.nan
(arr1==arr2).all() # error when one array is empty
d.equals(d2) #can check np.nan

#R:setdiff(x,y)
np.setdiff1d(arr1,arr2)
#R: intersect(x,y)
np.intersect1d(arr1,arr2)

#R: sum(v,na.rm=TRUE)
np.nansum(arr)     #all NA gives 0.0
s.sum(skipna=True) #all NA gives NA #default True


#tidy data, long/wide data
#dcast
#               index     columns
#R: dcast(long, subject ~ variable,value.var = "value")
d_wide = d.pivot(index='DATE',columns='jbar')[['aaa','bbb']]
d_wide.columns = d_wide.columns.droplevel() #multi-index to normal column
d_wide.reset_index(level =0, inplace=True)  #index to normal column

#https://stackoverflow.com/questions/34830597/pandas-melt-function
m = pd.melt(dhist, id_vars=['index'], var_name='Name')



#R: runMean(v,5) from TTR package
s = pd.Series(np.arange(10))
s.rolling(5).mean()

#groupby then sma
#https://stackoverflow.com/questions/13996302/python-rolling-functions-for-groupby-object
#R: d[, x_ema := runMean(x,5), by = y]
df.groupby("y")['x'].rolling(5).mean().reset_index(0,drop=True)
df.groupby(["y",'z'])['x'].rolling(5).mean().reset_index(drop=True) #groupby multiple index


#regression
#R: lm(y~x, data = d)
scipy.stats.linregress(x,y)
#https://stackoverflow.com/questions/31978948/python-stats-models-quadratic-term-in-regression
import statsmodels.formula.api as sm
model = sm.ols(formula = 'a ~ b + c + I(b**2)', data = data).fit()
model.summary()


#matplotlib (sorry, basically I use only ggplot no learning required)
#R: plot(1:5, 6:10)
import matplotlib.pyplot as plt
plt.plot(np.arange(5), np.arange(5,10))

#plot pdf
#R: pdf('a.pdf'); plot(1:5); dev.off()
#https://stackoverflow.com/questions/11328958/matplotlib-pyplot-save-the-plots-into-a-pdf

from matplotlib.backends.backend_pdf import PdfPages
pp = PdfPages("foo.pdf")
pp.savefig(plot1)
pp.savefig(plot2)
pp.savefig(plot3)
pp.close()


#ggplot 
#R: ggplot(d, aes(x=a,y=b))+geom_line()
from plotnine import *  #python ggplot package is no longer supported
ggplot(DF, aes(x = 'a', y = 'b'))+geom_line() #all col are string


#regression line
#R: ggplot(d, aes(x=a,y=b))+geom_line()+geom_smooth(method='lm', formula = "b~x")
from plotnine import *  
ggplot(DF, aes(x = 'a', y = 'b'))+geom_line()+stat_smooth(method='lm')

#geom_bar()
#https://stackoverflow.com/questions/47662234/plotnine-bar-plot-order-by-variable  
dimp['feat'] = pd.Categorical(dimp['feature'],  categories=dimp['feature'].values[::-1],  ordered=True)
ggplot(dimp, aes(x= 'feat',y='importance'))+geom_bar(stat = 'identity' , position = position_dodge())+coord_flip()

#plot to pdf
#R: pdf("a.pdf"); plot(p1);plot(p2); dev.off()
#R: p.save('a.pdf')

#add legend space in pdf https://github.com/has2k1/plotnine/issues/113
#+ theme(subplots_adjust={'right': 0.85})
#https://stackoverflow.com/questions/10101700/moving-matplotlib-legend-outside-of-the-axis-makes-it-cutoff-by-the-figure-box
from plotnine import *
from matplotlib.backends.backend_pdf import PdfPages
p1=ggplot(d, aes('a','b'))+geom_point()
p2=ggplot(d, aes('a','b'))+geom_line()
plot1 = p1.draw()
plot2 = p2.draw()

pp = PdfPages("foo.pdf")
pp.savefig(plot1, bbox_inches='tight') #similiar to tight_layout()
pp.savefig(plot2, bbox_inches='tight')
pp.close()

#wrap
def plot_to_pdf(list_p, fpath_pdf):
    '''
    list_p : list of ggplot
    '''
    #https://stackoverflow.com/questions/10101700/moving-matplotlib-legend-outside-of-the-axis-makes-it-cutoff-by-the-figure-box
    
    #from plotnine import *
    from matplotlib.backends.backend_pdf import PdfPages

    list_plot = []
    for p in list_p:
        iplot = p.draw()
        list_plot.append(iplot)

    pp = PdfPages(fpath_pdf)
    
    for iplot in list_plot:
        pp.savefig(iplot, bbox_inches='tight') #similiar to tight_layout()

    pp.close()




#R: ggplot(d, aes(x=a,y=b,color = p))+geom_point()+ facet_wrap(~q)
ggplot(aes(x='a',y='b',color='p'),data = d) +geom_point()+ facet_wrap('q')

ggplot(aes(x='a',y='b',color='p'),data = d) +geom_point()+facet_grid('q','r')

#do not use s.dt.date for plotting: use s.dt.normalize() 




#end
for 1==2:
    pass
    
 