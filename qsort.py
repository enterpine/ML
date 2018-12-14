a=[46,30,82,90,56,17,95,55,15,1,4,7,4]
#a=[15,30,17]
def qsort(a,start,end):
    if(len(a)<=1):
        return
    l = len(a)- 1
    i, j = start, end
    base = i
    cmp = j
    while(cmp!=base):
        if((a[cmp] < a[base] and cmp>base) or (a[cmp] > a[base] and cmp<base)):
            tmp = a[cmp]
            a[cmp] = a[base]
            a[base] = tmp
            tmpi = base
            base = cmp
            cmp = tmpi
        if(cmp<base):
            cmp=cmp+1
        else:
            cmp=cmp-1
    print(a[i:j+1],a[i:base],a[base+1:j+1],i,base-1,base+1,j)
    if(i<base-1 and i>=0 and base-1>=0 ):
        qsort(a,i,base-1)
    if(base+1<j and base+1 >=0 and j>=0):
        #print(a,base+1,j)
        qsort(a,base+1,j)

start=0
end = len(a)-1
qsort(a,start,end)
print(a)


