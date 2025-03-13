const double EPS=1e-8;
int sign(double val){
	if(fabs(val)<EPS) return 0;
	if(val>0) return 1;
	return -1;
}
int dcmp(double val1,double val2){
	return sign(val1-val2);
}
int gcd(int x,int y){return y ? gcd(y,x%y):x;}

//
//���µ������
//

struct Point{
	double x,y;
	Point(double a=0, double b=0) : x(a), y(b) {}
	friend Point operator + (Point a,Point b){
		return Point(a.x+b.x,a.y+b.y);
	}
	friend Point operator - (Point a,Point b){
		return Point(a.x-b.x,a.y-b.y);
	}
	friend Point operator / (Point a,double b){
		return Point(a.x/b,a.y/b);
	}
	friend Point operator * (Point a,double b){
		return Point(a.x*b,a.y*b);
	}
	friend Point operator * (double a,Point b){
		return Point(b.x*a,b.y*a);
	}
	friend double operator * (Point a,Point b){
		return a.x*b.x+a.y*b.y;
	}
	friend bool operator == (Point a,Point b){
		return (fabs(a.x-b.x)<EPS && fabs(a.y-b.y)<EPS);
	}
	friend bool operator != (Point a,Point b){
		return !(a==b);
	}
	//��%������,��Ϊ ^ ���ȼ�̫����
	friend double operator % (Point a,Point b){
		return a.x*b.y-a.y*b.x;
	}
	double ddis(){return x*x+y*y;}
	double dis(){return sqrt(x*x+y*y);}
	void read(){cin>>x>>y;}
	void print(char c='\n'){cout<<x<<" "<<y<<c;}
}INFP(1e20,1e20);

//�㵽��ľ���
double dis(Point a,Point b){
	return sqrt((a.x-b.x)*(a.x-b.x)+(a.y-b.y)*(a.y-b.y));
}

//�㵽��ľ���(������)
double ddis(Point a,Point b){
	return (a.x-b.x)*(a.x-b.x)+(a.y-b.y)*(a.y-b.y);
}

struct Seg{
	Point p1,p2;
	Seg(){}
	Seg(Point a,Point b) : p1(a),p2(b){}
	friend double operator * (Seg a,Seg b){
		return (a.p2-a.p1)*(b.p2-b.p1);
	}
	friend double operator % (Seg a,Seg b){
		return (a.p2-a.p1)%(b.p2-b.p1);
	}
	friend double operator % (Seg a,Point b){
		return (a.p2-a.p1)%(b-a.p1);
	}
	friend double operator % (Point a,Seg b){
		return (a-b.p1)%(b.p2-b.p1);
	}
	friend bool operator == (Seg a,Seg b){
		return (a.p1==b.p1 && a.p2==b.p2) || (a.p1==b.p2 && a.p2==b.p1);
	}
	void swap(){Point a=p2;p1=p2;p2=a;}
	double len(){return dis(p1,p2);}
	void read(){p1.read();p2.read();}
	void print(char c1='\n',char c2='\n'){p1.print(c1);p2.print(c2);}
};
struct Line{
	double A,B,C;
	Line(){A=B=C=0;}
	Line(double a,double b,double c) : A(a),B(b),C(c){}
	Line(Point p1,Point p2){
		A=p1.y-p2.y;
		B=p2.x-p1.x;
		C=p1.x*p2.y-p2.x*p1.y;
	}
	void read(){cin>>A>>B>>C;}
	void print(){cout<<A<<" "<<B<<" "<<C<<endl;}
};

//��������ʽֱ�߽���(������ʹ��)
Point calLineCross(Line L1,Line L2){
	auto [A1,B1,C1]=L1;
	auto [A2,B2,C2]=L2;
	double down=A1*B2-A2*B1;
	if(sign(down)!=0)
		return Point((B1*C2-B2*C1)/down,(A2*C1-A1*C2)/down);
	else{
		if(sign(A1)==0) return Point(-C2/A2,-C1/B1);
		else return Point(-C1/A1,-C2/B2);
	}
}

//�㵽�߶�orֱ�߾���,(ֱ��������)
double calPSDis1(Seg a,Point b){
	return fabs( (a.p2-a.p1) % (b-a.p1) ) / a.len();
}

//�㵽�߶ξ���,(����ֱ��������)
double calPSDis2(Seg a,Point b){
	int res1=sign((b-a.p1)*(a.p2-a.p1));
	int res2=sign((b-a.p2)*(a.p1-a.p2));
	if(res1>=0 && res2>=0) {
		return calPSDis1(a,b);
	}
	return min(dis(a.p1,b),dis(a.p2,b));
}

//��������ʽ��㵽ֱ�ߵľ���
double calPLDis(Line a,Point b){
	return fabs(a.A*b.x+a.B*b.y+a.C)/sqrt(a.A*a.A+a.B*a.B);
	//����fabs���Ա�ʾ���ֱ�ߵ�λ�ù�ϵ�����ǲ����Ǵ���0�������Ϸ� ����Ҫͬʱ��б������
}

//��������ʽ��㵽ֱ�ߵĴ���
Point calPLFeet(Line a,Point b){
	double x=(a.B*a.B*b.x-a.A*a.B*b.y-a.A*a.C)/(a.A*a.A+a.B*a.B);
	double y=(-a.A*a.B*b.x+a.A*a.A*b.y-a.B*a.C)/(a.A*a.A+a.B*a.B);
	return Point(x,y);
}

//��㵽�߶�orֱ�ߵĴ���
Point calPSFeet(Seg a,Point b){
	auto vec=a.p2-a.p1;
	return a.p1+vec*((b-a.p1)*vec)/(vec.x*vec.x+vec.y*vec.y);
}

//�����߶μ����ӳ��ߵĽ���
Point calSegCross(Seg a,Seg b){
	auto res1=(a.p2-a.p1)%(b.p1-a.p1);
	auto res2=(a.p2-a.p1)%(b.p2-a.p1);
	auto x=(res2*b.p1.x-res1*b.p2.x)/(res2-res1);
	auto y=(res2*b.p1.y-res1*b.p2.y)/(res2-res1);
	return Point(x,y);
}

//�жϵ��Ƿ����߶���(�����˵�)
bool isPAtS(Seg a,Point b){
	Point p1=a.p1-b,p2=a.p2-b;
	if(fabs(p1%p2)<EPS && sign(p1*p2)<=0) return 1;
	return 0;
}

//�ж��߶��Ƿ��ཻ
bool isSegCross(Seg a,Seg b) {
	double c1 = (a.p2 - a.p1)%(b.p1 - a.p1), c2 = (a.p2 - a.p1)%(b.p2 - a.p1);
	double c3 = (b.p2 - b.p1)%(a.p1 - b.p1), c4 = (b.p2 - b.p1)%(a.p2 - b.p1);
	//if�жϿ����Ƿ������߶��ڶ˵㴦�ཻ��������Ҫ���
	if (!sign(c1) || !sign(c2) || !sign(c3) || !sign(c4)) {
		bool f1 = isPAtS(a,b.p1);
		bool f2 = isPAtS(a,b.p2);
		bool f3 = isPAtS(b,a.p1);
		bool f4 = isPAtS(b,a.p2);
		bool f = (f1 | f2 | f3 | f4);
		return f;
	}
	return (sign(c1) * sign(c2) < 0 && sign(c3) * sign(c4) < 0);
}

//����ƽ��������ŷʽ����
double calPointMinDis(int l,int r,Point point[],int t[]){
	/*
	��ʼ�ݹ�ǰ�Ȱ���x����
	sort(point,point+n,[&](const Point& p1,const Point& p2){
		return p1.x<p2.x;
	});
	*/
    if(r-l==0)
        return 1e15;
    if(r-l==1)  //����ݹ����ֱ���������
        return dis(point[l],point[r]);
    int mid=(l+r)>>1;
    double ans=min(calPointMinDis(l,mid,point,t),calPointMinDis(mid+1,r,point,t));
    int cnt=0;
    for(int i=l;i<=r;i++)
        //����һ������Ǿ�����С������պ÷���mid����ans�����ڵĵ�
        if(point[i].x>=point[mid].x-ans && point[i].x<=point[mid].x+ans)
            t[++cnt]=i;
    sort(t+1,t+cnt+1,[&](int i,int j){
    	return point[i].y<point[j].y;
	}); //��y�����С����
    for(int i=1;i<=cnt;i++)
        for(int j=i+1;j<=cnt;j++){
            if(point[t[j]].y>=point[t[i]].y+ans) break;
			//������Ĵ�ֱ���볬��ans�Ͳ��ؼ����ˣ���Ȼ�����ܻ��Ϊ�µ�ans
            ans=min(ans,dis(point[t[i]],point[t[j]]));
        }
    return ans;
}

//
//���ϵ������
//

//
//����Բ���
//

struct Cir{
	Point o;
	double r;
	Cir(Point oo,double rr): o(oo),r(rr){}
	Cir(double x=0,double y=0,double rr=0){o.x=x,o.y=y,r=rr;}
	void read(){o.read();cin>>r;}
	void print(char c1='\n',char c2='\n'){
		o.print(c1);
		cout<<r<<c2;
	}
	Point getPoint(double theta){ //���ݼ��Ƿ���Բ��һ�������
	    double co=cos(theta)*r;
	    double si=sin(theta)*r;
	    if(sign(co)==0) co=0.0;
	    if(sign(si)==0) si=0.0;
	    return Point(o.x+co,o.y+si);
	}
	Point getPoint(Point p){ //����һ�㷵��Բ��һ�������
		auto vec=p-o;
		return o+vec*r/vec.dis();
	}
};

//�ж�Բ�ཻ���
int isCirCross(Cir a,Cir b){
	//����ֵǡ��Ϊ����Բ�Ĺ�����������
	int res=dcmp(dis(a.o,b.o),a.r+b.r);
	if(res>0) return 4;//����Բ���ཻ
	if(res==0) return 3;//����Բ���
	res=dcmp(dis(a.o,b.o),fabs(a.r-b.r));
	if(res>0) return 2;//����Բ�ཻ
	if(res==0) return 1;//��һ��Բ��������һ��Բ
	return 0;//һ��Բ����һ��Բ�ڲ�
}

//��������Բ�Ľ���
array<Point,2> calCirCross(Cir c1, Cir c2){
	Point vec12 = c2.o - c1.o; //��ԲԲ�ĵ�����
	double d = vec12.dis();    //Բ�ľ�
	double a = acos((c1.r * c1.r + d * d - c2.r * c2.r) / (2.0 * c1.r * d)); //vec12�루c1��һ�����㣩�ļн�
	double t = atan2(vec12.y, vec12.x); //vec12��x��ļн�
	// return {c1.o + polar(c1.r, t + a), c1.o + polar(c1.r, t - a)};
	return {c1.getPoint(t+a),c1.getPoint(t-a)};
}

//����������Բ
Cir calTrangleInnerCir(Point p1,Point p2,Point p3)
{
    Cir c;
    auto d1=dis(p1,p2),d2=dis(p1,p3),d3=dis(p2,p3);
    c.r=fabs((p2.x-p1.x)*(p3.y-p1.y)-(p2.y-p1.y)*(p3.x-p1.x))/(d1+d2+d3);
    c.o.x=(d3*p1.x+d1*p3.x+d2*p2.x)/(d1+d2+d3);
    c.o.y=(d3*p1.y+d1*p3.y+d2*p2.y)/(d1+d2+d3);
    return c;
}

//���������Բ
Cir calTrangleOuterCir(Point p1,Point p2,Point p3){
    Cir c;
    c.r=dis(p1,p2)*dis(p2,p3)*dis(p1,p3)/fabs((p2-p1)%(p3-p1))/2.0;
    auto a11=(p2.x-p1.x);
    auto a12=(p2.y-p1.y);
    auto b1=(p1.x*(p2.x-p1.x)+p1.y*(p2.y-p1.y)+0.5*ddis(p1,p2));
    auto a21=(p3.x-p1.x);
    auto a22=(p3.y-p1.y);
    auto b2=(p1.x*(p3.x-p1.x)+p1.y*(p3.y-p1.y)+0.5*ddis(p1,p3));
    auto down=(a11*a22-a21*a12);
    c.o.x=(b1*a22-b2*a12)/down;
    c.o.y=(a11*b2-b1*a21)/down;
    return c;
}

//ֱ�ߺ�Բ�������
int isSegCrossCir(Seg s,Cir c){
	int res=dcmp(calPSDis1(s,c.o),c.r);
	if(res==0) return 1;//����
	if(res==1) return 0;//���ཻ
	return 2;
}

//����ֱ����Բ����
array<Point,2> calSegCrossCir(Cir c,Seg s){
	auto feet=calPSFeet(s,c.o);
	int sta=isSegCrossCir(s,c);
	if(sta==0) return {INFP,INFP};
	if(sta==1) return {feet,feet};
	double base=sqrt(c.r*c.r-(feet-c.o).ddis());
	auto vec=s.p2-s.p1;
	Point e = vec/vec.dis();
	return {feet+e*base,feet-e*base};
}
array<Point,2> calPointCirTangent(Cir C,Point p) {
	auto d=dis(p,C.o);
    int aa = dcmp(d,C.r);
    if (aa<0) return {INFP,INFP};  //����Բ��
    else if (aa == 0)  return {p,p};//����Բ�ϣ��õ�����е�

    //����Բ�⣬�������е�
    double base = atan2( p.y - C.o.y, p.x - C.o.x );
    double ang = acos( C.r / d );
	return {C.getPoint( base - ang ),C.getPoint( base + ang )};
}

//����c1��c2������������c1�������е�
vector<Point>  calCirTangents(Cir c1,Cir c2){
	int CrossRes=isCirCross(c1,c2);
	vector<Point> res;
	auto d=dis(c1.o,c2.o);
    if(CrossRes==0) return res;
    double base = atan2(c2.o.y - c1.o.y, c2.o.x - c1.o.x); // AB�����ļ��ǣ�c1λ��Բ
    if(CrossRes==1){//����
    	res.push_back(c1.getPoint(base));
        return res;
    }
    double ang1 = acos((c1.r - c2.r) / d);//�⹫���ߵ�ͼ
    res.push_back(c1.getPoint(base + ang1));
    res.push_back(c1.getPoint(base - ang1));
    if(CrossRes==3)
        res.push_back(c1.getPoint(base));
		//һ���ڹ����ߣ���ʱc2�ϵĵ�Ϊ c2.getPoint(base + pi)
    else if(CrossRes==4) {//�����ڹ����ߣ���Ӧ�ڹ����ߵ�ͼ
        double ang2 = acos((c1.r + c2.r) / d);
        res.push_back(c1.getPoint(base + ang2));
    	res.push_back(c1.getPoint(base - ang2));
    }
    return res;
}

//����ε�����(��͹������)
Point calPolygonCentre (Point *point, int n) {
    double sum = 0.0, sumx = 0, sumy = 0;
    Point p1 = point[0], p2 = point[1], p3;
    for (int i = 2; i <= n-1; i++) {
        p3 = point[i];
        double area =  (p2-p1)%(p3-p2)/2.0;
        sum += area;
        sumx += (p1.x+p2.x+p3.x)*area;
        sumy += (p1.y+p2.y+p3.y)*area;
        p2 = p3;
    }
    return Point (sumx/(3.0*sum), sumy/(3.0*sum));
}

//
//����Բ���
//


//
//���¶�������
//

//�жϵ��Ƿ��ڶ����(���ܷǰ�)�ڲ�
int isPointInPolygon(int n,Point point[],Point P){
	int cnt=0;
	Point P1, P2;
	for (int i=0;i<n;i++){
		P1 = point[i]; P2 = point[(i+1)%n];
		if (isPAtS(Seg(P1,P2),P)) return 1;  // ������ڱ��� ���� 1
		if ((sign(P1.y - P.y) > 0 != sign(P2.y - P.y) > 0) &&
			sign(P.x - (P.y - P1.y) * (P1.x - P2.x) / (P1.y - P2.y) - P1.x) < 0)
			cnt++;
	}
	if (cnt&1) return 2;//���ڲ�����2
	else return 0;//���ⲿ����0
}

//��������(���Է�͹)
double calPolygonArea(Point point[], int n){
	double area = 0;
	for (int i = 0; i < n; i++)
		area +=point[i]%point[(i + 1) % n];
	return fabs(area) / 2.0;
}

//�������α�ֱ��ab�и�����
double calPolygonAreaCut(Point point[], int n,Point a,Point b){
	Point dir=b-a,tmp[110];
	int cnt=0;
	for(int i=0;i<n;i++){
		int res1=sign((point[i]-a)%dir);
		int res2=sign((point[(i+1)%n]-a)%dir);
		if(res1<=0) tmp[cnt++]=point[i];
		//����������������Ҳ�ĳ�>=0
		if(res1*res2==-1){
			tmp[cnt++]=calSegCross(Seg(a,b),Seg(point[i],point[(i+1)%n]));
		}
	}
	return calPolygonArea(tmp,cnt);
}

//pick����
array<int,2> pick(int n,Point point[]){
/*
	Pick �������������Ϊ����ļ򵥶���Σ�Ƥ�˶���˵����
	����� A ���ڲ������Ŀ i,���ϸ����Ŀ b �Ĺ�ϵ��
	A=i+b/2-1(û��ȡ��,������С��)
	ȡ�������ͼ�ε����Ϊһ��λ.��ƽ���ı��θ��,Ƥ�˶�����Ȼ������
	���������������θ�㣬Ƥ�˶������� A=2*i+b-2
*/
	int sum=0,num=0;
	for(int i=0;i<n;i++){
		int x=fabs(point[(i+1)%n].x-point[i].x)+0.5;
		int y=fabs(point[(i+1)%n].y-point[i].y)+0.5;
		num+=gcd(abs(x),abs(y));
		sum+=point[(i+1)%n]%point[i];
	}
	sum=abs(sum);//sumΪ���������
	return {(sum-num)/2+1,num};
}

//andrew��͹��
int andrew(int n,Point point[],Point res[]){
	//n>=2
	//point��res�±��0��ʼ,����ȡģ
	sort(point,point+n, [&](const Point& a,const Point& b){
		if(fabs(a.x-b.x)<EPS) return a.y<b.y;
		return a.x<b.x;
	});
	int m=0;
	for(int i=0;i<n;i++){
		while(m>1 && (res[m-1]-res[m-2])%(point[i]-res[m-2])<=0) --m;
		//����뱣��͹�����ϵĵ�,���԰� <= ���� <
		//͹��Ϊ��ʱ�뷽��
		res[m++]=point[i];
	}
	int k=m;
	for(int i=n-2;i>=0;i--){
		while(m>k && (res[m-1]-res[m-2])%(point[i]-res[m-2])<=0) --m;
		res[m++]=point[i];
	}
	--m;
	//point[1]���ظ���¼��,ɾȥ
	return m;
}

//��c���߶�ab�ľ���
double segdis(Point a,Point b,Point c){
	return fabs((b-a)%(c-a))/dis(a,b);
}

//��ת������͹��ֱ��
double rotate(int n,Point res[]){
	//��Ȼn>=2ʱ����Ϊ͹�����ڣ�������ת���ǵ�ʱ��Ҫ�� n>=3
	if(n==2) return dis(res[1],res[0]);
	int cur=0;
	double ans=0;
	for(int i=0;i<n;i++){
		while(segdis(res[i],res[(i+1)%n],res[cur])<=
				segdis(res[i],res[(i+1)%n],res[(cur+1)%n]))
			cur=(cur+1)%n;
		ans=max(ans,dis(res[i],res[cur]));
		ans=max(ans,dis(res[(i+1)%n],res[cur]));
	}
	return ans;
}

//�ж�͹��
bool isConvex(int n,Point point[]){
	//��������ķ�ʽΪ��ʱ������,�� < �ĳ� >
	//��Ϊ͹�����Ͽ����� �� ����� =
	for(int i=0;i<n;i++){
		if((point[i]-point[(i+1)%n])%(point[(i+2)%n]-point[(i+1)%n])>0)
			return 0;
	}
	return 1;
}

//
//���϶�������
//
