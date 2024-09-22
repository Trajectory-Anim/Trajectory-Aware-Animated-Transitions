// https://github.com/gorhill/Javascript-Voronoi
#include <iostream>
#include <vector>
#include <queue>
#include <cmath>
#include <algorithm>
#include <regex>
#include <iostream>
#include <fstream>
#include <random>
#define PI 3.14159265358979323846
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

#define RESOLUTION 1e2
#define ERROR 1e-6

class Site {
public:
    double x = 0;
    double y = 0;
    double r = 0;
    double w = 0;
    int id;
    Site(double x_in, double y_in) : x(x_in), y(y_in), r(0), w(0)
    {}
    Site()
    {}
    ~Site()
    {}
};

// Circle = CircleEvent
class Circle {
public:
    class Node* arc = nullptr;
    class Circle* rbLeft = nullptr;
    class Circle* rbRight = nullptr;
    class Circle* rbParent = nullptr;
    class Circle* rbNext = nullptr;
    class Circle* rbPrevious = nullptr;
    bool rbRed = false;
    class Site* site = nullptr;
    double x = 0;
    double y = 0;
    double ycenter = 0;
};
class Vertex {
public:
    double x;
    double y;
    Vertex(double x_in, double y_in) : x(x_in), y(y_in)
    {}
    Vertex()
    {}
    ~Vertex()
    {}
    static Vertex* createVertex(double x_in, double y_in) {
        Vertex* v = new Vertex(x_in, y_in);
        return v;
    }
};
class Edge {
public:
    class Site* lSite = nullptr;
    class Site* rSite = nullptr;
    class Vertex* va = nullptr;
    class Vertex* vb = nullptr;
    Edge(Site* start_in, Site* end_in) : lSite(start_in), rSite(end_in)
    {}
    Edge()
    {}
    ~Edge()
    {}
    static Edge* createEdge(Site* lSite, Site* rSite, Vertex* va = nullptr, Vertex* vb = nullptr) {
        Edge* e = new Edge(lSite, rSite);
        if (va) {
            e->setEdgeStartPoint(lSite, rSite, va);
        }
        if (vb) {
            e->setEdgeEndPoint(lSite, rSite, vb);
        }
        return e;
    }
    static Edge* createBorderEdge(Site* lSite, Vertex* va, Vertex* vb) {
        Edge* edge = new Edge(lSite, nullptr);
        edge->va = va;
        edge->vb = vb;
        return edge;
    }
    void setEdgeStartPoint(Site* lSite, Site* rSite, Vertex* vertex) {
        // std::cout << "setEdgeStartPoint" << std::endl;
        // std::cout << "vertex: " << vertex->x << " " << vertex->y << std::endl;
        if (this->va == nullptr && this->vb == nullptr) {
            // std::cout << "set va" << std::endl;
            this->va = vertex;
            this->lSite = lSite;
            this->rSite = rSite;
        }
        else if (this->lSite == rSite) {
            // std::cout << "set vb" << std::endl;
            this->vb = vertex;
        }
        else {
            // std::cout << "set va" << std::endl;
            this->va = vertex;
        }
    }
    void setEdgeEndPoint(Site* lSite, Site* rSite, Vertex* vertex) {
        this->setEdgeStartPoint(rSite, lSite, vertex);
    }
};
class HalfEdge {
public:
    class Site* site;
    class Edge* edge;
    double angle;
    HalfEdge()
    {}
    ~HalfEdge()
    {}
    static HalfEdge* createHalfEdge(Edge* edge, Site* lSite, Site* rSite) {
        HalfEdge* halfEdge = new HalfEdge();
        halfEdge->site = lSite;
        halfEdge->edge = edge;
        if (rSite) {
            halfEdge->angle = atan2(rSite->y - lSite->y, rSite->x - lSite->x);
        }
        else {
            Vertex* va = edge->va;
            Vertex* vb = edge->vb;
            if (edge->lSite == lSite) {
                halfEdge->angle = atan2(vb->x - va->x, va->y - vb->y);
            }
            else {
                halfEdge->angle = atan2(va->x - vb->x, vb->y - va->y);
            }
        }
        return halfEdge;
    }
    Vertex* getStartPoint() {
        return this->edge->lSite == this->site ? this->edge->va : this->edge->vb;
    }
    Vertex* getEndPoint() {
        return this->edge->lSite == this->site ? this->edge->vb : this->edge->va;
    }
};
class Cell {
public:
    int index;
    class Site* site = nullptr;
    std::vector<HalfEdge*> halfEdges;
    bool closeMe = false;
    Cell(Site* p_in) : site(p_in), halfEdges(std::vector<HalfEdge*>())
    {}
    Cell()
    {}
    ~Cell()
    {}
    static Cell* createCell(Site* site) {
        return new Cell(site);
    }
    int prepareHalfEdges() {
        int iHalfedge = halfEdges.size();
        Edge* edge;
        while (iHalfedge--) {
            edge = halfEdges[iHalfedge]->edge;
            if (edge->va == nullptr || edge->vb == nullptr) {
                halfEdges.erase(halfEdges.begin() + iHalfedge);
            }
        }
        sort(halfEdges.begin(), halfEdges.end(), [](HalfEdge* a, HalfEdge* b) {
            return  b->angle - a->angle<0;
            });
        return halfEdges.size();
    }
    std::vector<int> getNeighborIds() {
        std::vector<int> neighborIds = std::vector<int>();
        int iHalfedge = halfEdges.size();
        Edge* edge;
        while (iHalfedge--) {
            edge = halfEdges[iHalfedge]->edge;
            if (edge->lSite && edge->lSite->id != this->site->id) {
                neighborIds.push_back(edge->lSite->id);
            }
            else if (edge->rSite && edge->rSite->id != this->site->id) {
                neighborIds.push_back(edge->rSite->id);
            }
        }
        return neighborIds;
    }
    // std::vector<int> getBbox() {
    //     //TODO
    // }
    int pointIntersection(double x, double y) {
        int iHalfedge = halfEdges.size();
        HalfEdge* halfEdge;
        Vertex* p0;
        Vertex* p1;
        double r;
        while (iHalfedge--) {
            halfEdge = halfEdges[iHalfedge];
            p0 = halfEdge->getStartPoint();
            p1 = halfEdge->getEndPoint();
            r = (y - p0->y) * (p1->x - p0->x) - (x - p0->x) * (p1->y - p0->y);
            if (r * r < 1e-6) {
                return 0;
            }
            if (r > 0) {
                return -1;
            }
        }
        return 1;
    }
};

// Node = BeachSection
class Node {
public:
    Node* rbRight = nullptr;
    Node* rbLeft = nullptr;
    Node* rbNext = nullptr;
    Node* rbPrevious = nullptr;
    Node* rbParent = nullptr;
    bool rbRed = false;
    Site* site = nullptr;
    Circle* circleEvent = nullptr;
    Edge* edge = nullptr;
    static Node* createBeachSection(Site* site) {
        Node* beachsection = new Node();
        beachsection->site = site;
        return beachsection;
    }
};
template <class T>
class RBTree {
public:
    T* root = nullptr;
    RBTree(T* root_in) : root(root_in)
    {}
    RBTree()
    {}
    ~RBTree()
    {}
    void rbInsertSuccessor(T* node, T* successor) {
        T* parent;
        if (node) {
            successor->rbPrevious = node;
            successor->rbNext = node->rbNext;
            if (node->rbNext) {
                node->rbNext->rbPrevious = successor;
            }
            node->rbNext = successor;
            if (node->rbRight) {
                node = node->rbRight;
                while (node->rbLeft) {
                    node = node->rbLeft;
                }
                node->rbLeft = successor;
            }
            else {
                node->rbRight = successor;
            }
            parent = node;
        }
        else if (this->root) {
            node = this->getFirst(this->root);
            successor->rbPrevious = nullptr;
            successor->rbNext = node;
            node->rbPrevious = successor;
            node->rbLeft = successor;
            parent = node;
        }
        else {
            successor->rbPrevious = nullptr;
            successor->rbNext = nullptr;
            this->root = successor;
            parent = nullptr;
        }
        successor->rbLeft = nullptr;
        successor->rbRight = nullptr;
        successor->rbRed = true;
        successor->rbParent = parent;

        T* grandpa, * uncle;
        node = successor;
        while (parent && parent->rbRed) {
            grandpa = parent->rbParent;
            if (parent == grandpa->rbLeft) {
                uncle = grandpa->rbRight;
                if (uncle && uncle->rbRed) {
                    parent->rbRed = false;
                    uncle->rbRed = false;
                    grandpa->rbRed = true;
                    node = grandpa;
                }
                else {
                    if (node == parent->rbRight) {
                        this->rbRotateLeft(parent);
                        node = parent;
                        parent = node->rbParent;
                    }
                    parent->rbRed = false;
                    grandpa->rbRed = true;
                    this->rbRotateRight(grandpa);
                }
            }
            else {
                uncle = grandpa->rbLeft;
                if (uncle && uncle->rbRed) {
                    parent->rbRed = false;
                    uncle->rbRed = false;
                    grandpa->rbRed = true;
                    node = grandpa;
                }
                else {
                    if (node == parent->rbLeft) {
                        this->rbRotateRight(parent);
                        node = parent;
                        parent = node->rbParent;
                    }
                    parent->rbRed = false;
                    grandpa->rbRed = true;
                    this->rbRotateLeft(grandpa);
                }
            }
            parent = node->rbParent;
        }
        this->root->rbRed = false;
    }
    void rbRemoveNode(T* node) {
        if (node == nullptr) {
            std::cout << "nullptr" << std::endl;
        }
        if (node->rbNext) {
            node->rbNext->rbPrevious = node->rbPrevious;
        }
        if (node->rbPrevious) {
            node->rbPrevious->rbNext = node->rbNext;
        }
        node->rbNext = nullptr;
        node->rbPrevious = nullptr;
        T* parent, * left, * right, * next;
        parent = node->rbParent;
        left = node->rbLeft;
        right = node->rbRight;
        next = nullptr;
        if (left == nullptr) {
            next = right;
        }
        else if (right == nullptr) {
            next = left;
        }
        else {
            next = this->getFirst(right);
        }
        if (parent) {
            if (parent->rbLeft == node) {
                parent->rbLeft = next;
            }
            else {
                parent->rbRight = next;
            }
        }
        else {
            this->root = next;
        }
        bool isRed;
        if (left && right) {
            isRed = next->rbRed;
            next->rbRed = node->rbRed;
            next->rbLeft = left;
            left->rbParent = next;
            if (next != right) {
                parent = next->rbParent;
                next->rbParent = node->rbParent;
                node = next->rbRight;
                parent->rbLeft = node;
                next->rbRight = right;
                right->rbParent = next;
            }
            else {
                next->rbParent = parent;
                parent = next;
                node = next->rbRight;
            }
        }
        else {
            isRed = node->rbRed;
            node = next;
        }
        if (node) {
            node->rbParent = parent;
        }
        if (isRed) {
            return;
        }
        if (node && node->rbRed) {
            node->rbRed = false;
            return;
        }
        T* sibling;
        do {
            if (node == this->root)break;
            if (node == parent->rbLeft) {
                sibling = parent->rbRight;
                if (sibling->rbRed) {
                    sibling->rbRed = false;
                    parent->rbRed = true;
                    this->rbRotateLeft(parent);
                    sibling = parent->rbRight;
                }
                if ((sibling->rbLeft && sibling->rbLeft->rbRed) || (sibling->rbRight && sibling->rbRight->rbRed)) {
                    if (!sibling->rbRight || !sibling->rbRight->rbRed) {
                        sibling->rbLeft->rbRed = false;
                        sibling->rbRed = true;
                        this->rbRotateRight(sibling);
                        sibling = parent->rbRight;
                    }
                    sibling->rbRed = parent->rbRed;
                    parent->rbRed = false;
                    if (sibling->rbRight) {
                        sibling->rbRight->rbRed = false;
                    }
                    this->rbRotateLeft(parent);
                    node = this->root;
                    break;
                }
            }
            else {
                sibling = parent->rbLeft;
                if (sibling->rbRed) {
                    sibling->rbRed = false;
                    parent->rbRed = true;
                    this->rbRotateRight(parent);
                    sibling = parent->rbLeft;
                }
                if ((sibling->rbLeft && sibling->rbLeft->rbRed) || (sibling->rbRight && sibling->rbRight->rbRed)) {
                    if (!sibling->rbLeft || !sibling->rbLeft->rbRed) {
                        sibling->rbRight->rbRed = false;
                        sibling->rbRed = true;
                        this->rbRotateLeft(sibling);
                        sibling = parent->rbLeft;
                    }
                    sibling->rbRed = parent->rbRed;
                    parent->rbRed = false;
                    if (sibling->rbLeft) {
                        sibling->rbLeft->rbRed = false;
                    }
                    this->rbRotateRight(parent);
                    node = this->root;
                    break;
                }
            }
            sibling->rbRed = true;
            node = parent;
            parent = node->rbParent;
        } while (!node->rbRed);
        if (node) {
            node->rbRed = false;
        }
        //     if (node == this->root) {
        //         break;
        //     }
        //     if (node == parent->rbLeft) {
        //         sibling = parent->rbRight;
        //         if (sibling->rbRed) {
        //             sibling->rbRed = false;
        //             parent->rbRed = true;
        //             this->rbRotateLeft(parent);
        //             sibling = parent->rbRight;
        //         }
        //         if ((!sibling->rbLeft || !sibling->rbLeft->rbRed) &&
        //             (!sibling->rbRight || !sibling->rbRight->rbRed)) {
        //             sibling->rbRed = true;
        //             node = parent;
        //             parent = node->rbParent;
        //             continue;
        //         }
        //         if (!sibling->rbRight || !sibling->rbRight->rbRed) {
        //             sibling->rbLeft->rbRed = false;
        //             sibling->rbRed = true;
        //             this->rbRotateRight(sibling);
        //             sibling = parent->rbRight;
        //         }
        //         sibling->rbRed = parent->rbRed;
        //         parent->rbRed = false;
        //         sibling->rbRight->rbRed = false;
        //         this->rbRotateLeft(parent);
        //         node = this->root;
        //         break;
        //     }
        //     else {
        //         sibling = parent->rbLeft;
        //         if (sibling->rbRed) {
        //             sibling->rbRed = false;
        //             parent->rbRed = true;
        //             this->rbRotateRight(parent);
        //             sibling = parent->rbLeft;
        //         }
        //         if ((!sibling->rbLeft || !sibling->rbLeft->rbRed) &&
        //             (!sibling->rbRight || !sibling->rbRight->rbRed)) {
        //             sibling->rbRed = true;
        //             node = parent;
        //             parent = node->rbParent;
        //             continue;
        //         }
        //         if (!sibling->rbLeft || !sibling->rbLeft->rbRed) {
        //             sibling->rbRight->rbRed = false;
        //             sibling->rbRed = true;
        //             this->rbRotateLeft(sibling);
        //             sibling = parent->rbLeft;
        //         }
        //         sibling->rbRed = parent->rbRed;
        //         parent->rbRed = false;
        //         sibling->rbLeft->rbRed = false;
        //         this->rbRotateRight(parent);
        //         node = this->root;
        //         break;
        //         // if ((sibling->rbLeft && sibling->rbLeft->rbRed) || (sibling->rbRight && sibling->rbRight->rbRed)) {
        //         //     if (sibling->rbLeft == nullptr || !sibling->rbLeft->rbRed) {
        //         //         sibling->rbRight->rbRed = false;
        //         //         sibling->rbRed = true;
        //         //         this->rbRotateLeft(sibling);
        //         //         sibling = parent->rbLeft;
        //         //     }
        //         //     sibling->rbRed = parent->rbRed;
        //         //     parent->rbRed = false;
        //         //     if (sibling->rbLeft) {
        //         //         sibling->rbLeft->rbRed = false;
        //         //     }
        //         //     this->rbRotateRight(parent);
        //         //     node = this->root;
        //         //     break;
        //         // }
        //     }
        // } while (!node->rbRed);
        // if (node) {
        //     node->rbRed = false;
        // }
    }
    void rbRotateLeft(T* node) {
        T* p = node;
        T* q = node->rbRight;
        T* parent = p->rbParent;
        if (parent) {
            if (parent->rbLeft == p) {
                parent->rbLeft = q;
            }
            else {
                parent->rbRight = q;
            }
        }
        else {
            this->root = q;
        }
        q->rbParent = parent;
        p->rbParent = q;
        p->rbRight = q->rbLeft;
        if (p->rbRight) {
            p->rbRight->rbParent = p;
        }
        q->rbLeft = p;
    }
    void rbRotateRight(T* node) {
        T* p = node;
        T* q = node->rbLeft;
        T* parent = p->rbParent;
        if (parent) {
            if (parent->rbLeft == p) {
                parent->rbLeft = q;
            }
            else {
                parent->rbRight = q;
            }
        }
        else {
            this->root = q;
        }
        q->rbParent = parent;
        p->rbParent = q;
        p->rbLeft = q->rbRight;
        if (p->rbLeft) {
            p->rbLeft->rbParent = p;
        }
        q->rbRight = p;
    }
    T* getFirst(T* node) {
        while (node->rbLeft) {
            node = node->rbLeft;
        }
        return node;
    }
    T* getLast(T* node) {
        while (node->rbRight) {
            node = node->rbRight;
        }
        return node;
    }
};

class Bbox {
public:
    double xl, xr, yt, yb;
};

class Voronoi {
public:
    std::vector<Site*> sites;
    std::vector<Cell*> cells;
    std::vector<Edge*> edges;
    std::vector<Vertex*> vertices;
    double epsilon = 1e-9;
    double sweepline = 0;
    RBTree<Node> beachline;
    RBTree<Circle> circleEvents;
    Circle* firstCircleEvent;
    Bbox bbox;
    std::vector<std::vector<double>> convex_hull_boundary;
    std::vector<std::tuple<double, double, double>> convex_hull_boundary_edges;

    std::vector<double> capacity;
    std::vector<std::pair<std::vector<std::vector<double>>, std::vector<int>>> resultCellsFlowers;

    double margin;
    Voronoi() {
    }
    ~Voronoi()
    {}
//    void addSite(Site* s) {
//        sites.push_back(s);
//    }
//    void addCell(Cell* c) {
//        cells.push_back(c);
//    }
    void reset() {
        this->beachline = RBTree<Node>();
        this->beachline.root = nullptr;
        this->circleEvents = RBTree<Circle>();
        this->circleEvents.root = nullptr;
        this->firstCircleEvent = nullptr;
        // this->sites = vector<Site*>();
        for (auto site: this->sites) {
            site->id = 0;
        }
        this->cells = std::vector<Cell*>();
        this->edges = std::vector<Edge*>();
        this->vertices = std::vector<Vertex*>();
        this->bbox = Bbox();
        this->bbox.xl = 0.0;
        this->bbox.xr = 1.0 * RESOLUTION;
        this->bbox.yt = 0.0;
        this->bbox.yb = 1.0 * RESOLUTION;
        // tolerance
    }

    void clearSites() {
        for (auto site: this->sites) {
            delete site;
        }
        this->sites = std::vector<Site*>();
    }

    double leftBreakPoint(Node* arc, double directrix) {
//        Site* site = arc->site;
//        double rfocx = site->x;
//        double rfocy = site->y;
//        double pby2 = rfocy - directrix;
//        double x1 = site->x;
//        double y1 = site->y;
//        double k1 = (y1 + directrix) / 2;
//        double p1 = k1 - directrix;
//        if (abs(pby2) < 1e-7) {
//            return rfocx;
//        }
//        Node* lArc = arc->rbPrevious;
//        if (lArc == nullptr) {
//            return -std::numeric_limits<double>::infinity();
//        }
//        site = lArc->site;
//        double lfocx = site->x;
//        double lfocy = site->y;
//        double plby2 = lfocy - directrix;
//        double x2 = site->x;
//        double y2 = site->y;
//        double k2 = (y2 + directrix) / 2;
//        double p2 = k2 - directrix;
//        if (abs(plby2) < 1e-7) {
//            return lfocx;
//        }
//        double hl = lfocx - rfocx;
//        double aby2 = 1 / pby2 - 1 / plby2;
//        double b = hl / plby2;
//        if (abs(aby2) > 1e-7) {
//            double a = 1/(4*p2) - 1/(4*p1);
//            double bb = (x1-x2) / (2*p2);
//            double c = (x2-x1)*(x2-x1) / (4*p2) + (k2-k1);
//            std::cout << x2 << '/' << (-b + sqrt(b * b - 2 * aby2 * (hl * hl / (-2 * plby2) - lfocy + plby2 / 2 + rfocy - pby2 / 2))) / aby2 + rfocx << '/' << x1
//                      << " " << (-bb - sqrt(bb*bb - 4*a*c)) / (2*a) + x1 <<  " " << (-bb + sqrt(bb*bb - 4*a*c)) / (2*a) + x1 << " check " << std::endl;
//            return (-b + sqrt(b * b - 2 * aby2 * (hl * hl / (-2 * plby2) - lfocy + plby2 / 2 + rfocy - pby2 / 2))) / aby2 + rfocx;
//        }
//        return (rfocx + lfocx) / 2;
        Site* site = arc->site;
        double x1 = site->x;
        double y1 = site->y;
        double r1 = site->r;
        double k1 = (y1 + directrix) / 2;
        double p1 = k1 - directrix;
        if (abs(p1) < 1e-7) {
            return x1;
        }
        Node* lArc = arc->rbPrevious;
        if (lArc == nullptr) {
            return -std::numeric_limits<double>::infinity();
        }
        site = lArc->site;
        double x2 = site->x;
        double y2 = site->y;
        double r2 = site->r;
        double k2 = (y2 + directrix) / 2;
        double p2 = k2 - directrix;
        if (abs(p2) < 1e-7) {
            return x2;
        }
        double a = (1/p2-1/p1) / 4;
        double b = - (x2-x1) / (2*p2);
        double c = (x2-x1)*(x2-x1) / (4*p2) + (k2-k1) - (r2*r2/(4*p2) - (r1*r1) / (4*p1));
        double llbp = leftBreakPoint(arc->rbPrevious, directrix);
        if (abs(p1-p2)>1e-7) {
//            std::cout << "checkLBP " << x1 << " " << x2 << " " << std::endl;
//            std::cout << "checkLBP " << y1 << " " << y2 << " " << std::endl;
//            std::cout << "checkLBP " << (-b - sqrt(b*b - 4*a*c)) / (2*a) + x1 << " " << (-b + sqrt(b*b - 4*a*c)) / (2*a) + x1 << std::endl;
            double lbp1 = (-b - sqrt(b*b - 4*a*c)) / (2*a) + x1;
            double lbp2 = (-b + sqrt(b*b - 4*a*c)) / (2*a) + x1;
            if (lbp1>llbp) {
                return lbp1;
            } else {
                return lbp2;
            }
            return (lbp1 < lbp2) ? lbp1 : lbp2;

        }
        // std::cout << "other sol" << std::endl;
        double sol =  (x1 + x2) / 2 + (y2*y2-y1*y1-r2*r2+r1*r1) / (2*(x2-x1));
        if (sol > llbp) {
            return sol;
        } else {
            return -std::numeric_limits<double>::infinity();
        }
    }
    double rightBreakPoint(Node* arc, double directrix) {
        Node* rArc = arc->rbNext;
        if (rArc) {
            double res = leftBreakPoint(rArc, directrix);
            if (res != -std::numeric_limits<double>::infinity()) {
                return res;
            }
            else {
                return std::numeric_limits<double>::infinity();
            }
        }
        Site* site = arc->site;
        return abs((site->y + site->r) - directrix) < 1e-7 ? site->x : std::numeric_limits<double>::infinity();
    }

    void detachBeachSection(Node* beachsection) {
        this->detachCircleEvent(beachsection);
        if (!beachsection)std::cout << "beachsection" << std::endl;
        this->beachline.rbRemoveNode(beachsection);
    }
    void removeBeachSection(Node* beachsection) {
        Circle* circle = beachsection->circleEvent;
        double x = circle->x;
        double y = circle->ycenter;
        Vertex* vertex = Vertex::createVertex(x, y);
        Node* previous = beachsection->rbPrevious;
        Node* next = beachsection->rbNext;
        std::vector<Node*> disappearingTransitions = std::vector<Node*>();
        disappearingTransitions.push_back(beachsection);
        this->detachBeachSection(beachsection);

        Node* lArc = previous;
        while (lArc->circleEvent && abs(x - lArc->circleEvent->x) < 1e-9 && abs(y - lArc->circleEvent->ycenter) < 1e-9) {
            previous = lArc->rbPrevious;
            disappearingTransitions.insert(disappearingTransitions.begin(), lArc);
            this->detachBeachSection(lArc);
            lArc = previous;
            // std::cout << "detaching" << std::endl;
        }
        disappearingTransitions.insert(disappearingTransitions.begin(), lArc);
        this->detachCircleEvent(lArc);

        Node* rArc = next;
        while (rArc->circleEvent && abs(x - rArc->circleEvent->x) < 1e-9 && abs(y - rArc->circleEvent->ycenter) < 1e-9) {
            next = rArc->rbNext;
            disappearingTransitions.push_back(rArc);
            this->detachBeachSection(rArc);
            rArc = next;
            // std::cout << "detaching" << std::endl;
        }
        disappearingTransitions.push_back(rArc);
        this->detachCircleEvent(rArc);

        int nArcs = disappearingTransitions.size();
        for (int iArc = 1; iArc < nArcs; iArc++) {
            rArc = disappearingTransitions[iArc];
            lArc = disappearingTransitions[iArc - 1];
            rArc->edge->setEdgeStartPoint(lArc->site, rArc->site, vertex);
        }
        lArc = disappearingTransitions[0];
        rArc = disappearingTransitions[nArcs - 1];
        // std::cout << "create edge during remove beachsection" << std::endl;
        // std::cout << "lArc" << lArc->site->x << " " << lArc->site->y << std::endl;
        // std::cout << "rArc" << rArc->site->x << " " << rArc->site->y << std::endl;
        // std::cout << "vertex" << vertex->x << " " << vertex->y << std::endl;
        rArc->edge = Edge::createEdge(lArc->site, rArc->site, nullptr, vertex);
        Site* lSite = lArc->site;
        Site* rSite = rArc->site;
        this->edges.push_back(rArc->edge);
        this->cells[lSite->id]->halfEdges.push_back(HalfEdge::createHalfEdge(rArc->edge, lSite, rSite));
        this->cells[rSite->id]->halfEdges.push_back(HalfEdge::createHalfEdge(rArc->edge, rSite, lSite));


        this->attachCircleEvent(lArc);
        this->attachCircleEvent(rArc);
    }
    void addBeachSection(Site* site) {
        double x = site->x;
        double directrix = site->y;
        double xl,xr;
        xl = site->x-site->r;
        xr = site->x+site->r;
        double dxl, dxr;
        Node* lArc = nullptr;
        Node* rArc = nullptr;
        Node* node = this->beachline.root;
        while (node) {
            dxl = this->leftBreakPoint(node, directrix) - xl;
            if (dxl > 1e-9) {
                node = node->rbLeft;
            }
            else {
                dxr = xl - this->rightBreakPoint(node, directrix);
                if (dxr > -1e-9) {
                    if (!node->rbRight) {
                        lArc = node;
                        break;
                    }
                    node = node->rbRight;
                }
                else {
                    if (dxl > -1e-9) {
                        lArc = node->rbPrevious;
                        // rArc = node;
                    }
                    else if (dxr > -1e-9) {
                        lArc = node;
                        // rArc = node->rbNext;
                    }
                    else {
                        lArc = node;
                        // lArc = rArc = node;
                    }
                    break;
                }
            }
        }
        node = this->beachline.root;
        while (node) {
            dxl = this->leftBreakPoint(node, directrix) - xr;
            if (dxl > 1e-9) {
                node = node->rbLeft;
            }
            else {
                dxr = xr - this->rightBreakPoint(node, directrix);
                if (dxr > -1e-9) {
                    if (!node->rbRight) {
                        // lArc = node;
                        break;
                    }
                    node = node->rbRight;
                }
                else {
                    if (dxl > -1e-9) {
                        // lArc = node->rbPrevious;
                        rArc = node;
                    }
                    else if (dxr > -1e-9) {
                        // lArc = node;
                        rArc = node->rbNext;
                    }
                    else {
                        rArc = node;
                        // lArc = rArc = node;
                    }
                    break;
                }
            }
        }
        Node* newArc = Node::createBeachSection(site);
        this->beachline.rbInsertSuccessor(lArc, newArc);
        if (lArc == nullptr && rArc == nullptr) {
            return;
        }
        if (lArc == rArc) {
            this->detachCircleEvent(lArc);

            rArc = Node::createBeachSection(lArc->site);
            this->beachline.rbInsertSuccessor(newArc, rArc);
            // std::cout << "create edge during add beachsection" << std::endl;
            // std::cout << "lArc" << lArc->site->x << " " << lArc->site->y << std::endl;
            // std::cout << "newArc" << newArc->site->x << " " << newArc->site->y << std::endl;
            newArc->edge = rArc->edge = Edge::createEdge(lArc->site, newArc->site);
            this->edges.push_back(newArc->edge);
            Site* lSite = lArc->site;
            Site* rSite = newArc->site;
            this->cells[lSite->id]->halfEdges.push_back(HalfEdge::createHalfEdge(newArc->edge, lSite, rSite));
            this->cells[rSite->id]->halfEdges.push_back(HalfEdge::createHalfEdge(newArc->edge, rSite, lSite));
            this->attachCircleEvent(lArc);
            this->attachCircleEvent(rArc);
            return;
        }
        if (lArc && rArc == nullptr) {
            newArc->edge = Edge::createEdge(lArc->site, newArc->site);
            this->edges.push_back(newArc->edge);
            Site* lSite = lArc->site;
            Site* rSite = newArc->site;
            this->cells[lSite->id]->halfEdges.push_back(HalfEdge::createHalfEdge(newArc->edge, lSite, rSite));
            this->cells[rSite->id]->halfEdges.push_back(HalfEdge::createHalfEdge(newArc->edge, rSite, lSite));
            return;
        }
        if (lArc != rArc) {
            this->detachCircleEvent(lArc);
            this->detachCircleEvent(rArc);
            // deal with all transitions [lArc, rArc]
            Node* tlArc = lArc;
            Node* trArc = lArc->rbNext->rbNext; // because lArc->rbNext is newArc
            std::vector<Vertex*> disappear_vertices;
            while(tlArc && trArc) {
                Site* lSite = tlArc->site;
                Site* rSite = trArc->site;
                Site* site = newArc->site;
                double bx = site->x;
                double by = site->y;
                double ax = lSite->x - bx;
                double ay = lSite->y - by;
                double cx = rSite->x - bx;
                double cy = rSite->y - by;
                double ar = lSite->r, br = site->r, cr = rSite->r;
                double d = 2 * (ax * cy - ay * cx);
                double ha = ax * ax + ay * ay;
                double hc = cx * cx + cy * cy;
                double sqrt_ha = sqrt(ha);
                double sqrt_hc = sqrt(hc);
                double oa = (br * br - ar * ar + ha) / (2 * sqrt_ha);
                double oc = (br * br - cr * cr + hc) / (2 * sqrt_hc);
                double la = - ax * (oa / sqrt_ha * ax) - ay * (oa / sqrt_ha * ay);
                double lc = - cx * (oc / sqrt_hc * cx) - cy * (oc / sqrt_hc * cy);
                double x = (ay * lc - cy * la) / (d/2);
                double y = (- ax * lc + cx * la) / (d/2);
                double vx = x + bx;
                // double vy = y + by + sqrt(x * x + y * y - br * br);
                double vy = y + by;
                // std::cout << vx << " " << vy << " with r" << std::endl;
                Vertex* vertex = Vertex::createVertex(vx, vy);
                disappear_vertices.push_back(vertex);
                if (d >= -1e-3) {
                    std::swap(lSite, rSite);
                }
                if (trArc->edge->va!=nullptr) {
                    if (trArc->edge->lSite==lSite){
                        trArc->edge->setEdgeEndPoint(lSite, rSite, vertex);
                    }
                    else{
                        trArc->edge->setEdgeEndPoint(rSite, lSite, vertex);
                    }
                }
                else {
                    trArc->edge->setEdgeStartPoint(rSite, lSite, vertex);
                }
                if (disappear_vertices.size()>1){
                    Vertex* v = disappear_vertices[disappear_vertices.size()-2];
                    Edge *e = Edge::createEdge(tlArc->site, site, v, vertex);
                    this->edges.push_back(e);
                    this->cells[tlArc->site->id]->halfEdges.push_back(HalfEdge::createHalfEdge(e, tlArc->site, site));
                    this->cells[site->id]->halfEdges.push_back(HalfEdge::createHalfEdge(e, site, tlArc->site));
                }
                if (trArc == rArc) {
                    break;
                }
                tlArc = trArc;
                trArc = trArc->rbNext;
            }

            newArc->edge = Edge::createEdge(lArc->site, newArc->site, nullptr, disappear_vertices[0]);
            this->edges.push_back(newArc->edge);
            this->cells[lArc->site->id]->halfEdges.push_back(HalfEdge::createHalfEdge(newArc->edge, lArc->site, newArc->site));
            this->cells[newArc->site->id]->halfEdges.push_back(HalfEdge::createHalfEdge(newArc->edge, newArc->site, lArc->site));

            rArc->edge = Edge::createEdge(newArc->site, rArc->site,nullptr, disappear_vertices[disappear_vertices.size()-1]);
            this->edges.push_back(rArc->edge);
            this->cells[newArc->site->id]->halfEdges.push_back(HalfEdge::createHalfEdge(rArc->edge, newArc->site, rArc->site));
            this->cells[rArc->site->id]->halfEdges.push_back(HalfEdge::createHalfEdge(rArc->edge, rArc->site, newArc->site));

            // // TODO add arc edges
            // Node* now = lArc->rbNext->rbNext;
            // // create vertex for lArc, lArc->rbNext->rbNext, newArc
            // Site* lSite = lArc->site;
            // Site* rSite = newArc->site;
            // Site* bSite = lArc->rbNext->rbNext->site;
            // double bx = bSite->x;
            // double by = bSite->y;
            // double ax = lSite->x - bx;
            // double ay = lSite->y - by;
            // double cx = rSite->x - bx;
            // double cy = rSite->y - by;
            // double ar = lSite->r, br = bSite->r, cr = rSite->r;
            // double d = 2 * (ax * cy - ay * cx);
            // double ha = ax * ax + ay * ay;
            // double hc = cx * cx + cy * cy;
            // double sqrt_ha = sqrt(ha);
            // double sqrt_hc = sqrt(hc);
            // double oa = (br * br - ar * ar + ha) / (2 * sqrt_ha);
            // double oc = (br * br - cr * cr + hc) / (2 * sqrt_hc);
            // double la = - ax * (oa / sqrt_ha * ax) - ay * (oa / sqrt_ha * ay);
            // double lc = - cx * (oc / sqrt_hc * cx) - cy * (oc / sqrt_hc * cy);
            // double x = (ay * lc - cy * la) / (d/2);
            // double y = (- ax * lc + cx * la) / (d/2);
            // double vx = x + bx;
            // // double vy = y + by + sqrt(x * x + y * y - br * br);
            // double vy = y + by;
            // std::cout << vx << " " << vy << " with r" << std::endl;
            // Vertex* vertex = Vertex::createVertex(vx, vy);
            // Node* tArc = lArc->rbNext->rbNext;
            // if (tArc->edge->lSite==lSite && tArc->edge->va!=nullptr) {
            //     tArc->edge->setEdgeEndPoint(lSite, bSite, vertex);
            // }
            // else {
            //     tArc->edge->setEdgeStartPoint(lSite, bSite, vertex);
            // }
            // newArc->edge = Edge::createEdge(lSite, newArc->site, nullptr, vertex);
            // this->edges.push_back(newArc->edge);
            // this->cells[lSite->id]->halfEdges.push_back(HalfEdge::createHalfEdge(newArc->edge, lSite, newArc->site));
            // this->cells[newArc->site->id]->halfEdges.push_back(HalfEdge::createHalfEdge(newArc->edge, newArc->site, lSite));
            // // newArc->edge->setEdgeEndPoint(lSite, newArc->site, vertex);
            // Node* newrArc = Node::createBeachSection(rArc->site);
            // if (tArc == rArc){
            //     this->beachline.rbInsertSuccessor(rArc->rbPrevious, newrArc);
            //     this->detachBeachSection(rArc);
            //     newrArc->edge = Edge::createEdge(newArc->site, rArc->site, nullptr, vertex);
            //     this->edges.push_back(newrArc->edge);
            //     this->cells[newArc->site->id]->halfEdges.push_back(HalfEdge::createHalfEdge(newrArc->edge, newArc->site, rArc->site));
            //     this->cells[rArc->site->id]->halfEdges.push_back(HalfEdge::createHalfEdge(newrArc->edge, rArc->site, newArc->site));
            // }
            // // this->detachBeachSection(tArc);
            // // TODO

            // // create vertex for newArc, rArc->rbPrevious, rArc
            // if (rArc->rbPrevious != newArc && rArc != tArc){
            //     lSite = newArc->site;
            //     rSite = rArc->site;
            //     bSite = rArc->rbPrevious->site;
            //     bx = bSite->x;
            //     by = bSite->y;
            //     ax = lSite->x - bx;
            //     ay = lSite->y - by;
            //     cx = rSite->x - bx;
            //     cy = rSite->y - by;
            //     ar = lSite->r, br = bSite->r, cr = rSite->r;
            //     d = 2 * (ax * cy - ay * cx);
            //     ha = ax * ax + ay * ay;
            //     hc = cx * cx + cy * cy;
            //     sqrt_ha = sqrt(ha);
            //     sqrt_hc = sqrt(hc);
            //     oa = (br * br - ar * ar + ha) / (2 * sqrt_ha);
            //     oc = (br * br - cr * cr + hc) / (2 * sqrt_hc);
            //     la = - ax * (oa / sqrt_ha * ax) - ay * (oa / sqrt_ha * ay);
            //     lc = - cx * (oc / sqrt_hc * cx) - cy * (oc / sqrt_hc * cy);
            //     x = (ay * lc - cy * la) / (d/2);
            //     y = (- ax * lc + cx * la) / (d/2);
            //     vx = x + bx;
            //     // vy = y + by + sqrt(x * x + y * y - br * br);
            //     vy = y + by;
            //     std::cout << vx << " " << vy << " with r" << std::endl;
            //     vertex = Vertex::createVertex(vx, vy);
            //     // rArc->edge->setEdgeStartPoint(lSite, bSite, vertex);
            //     // rArc->rbPrevious->edge->setEdgeStartPoint(lSite, rSite, vertex);
            //     tArc = rArc;
            //     if (tArc->edge->lSite==lSite && tArc->edge->va!=nullptr) {
            //         tArc->edge->setEdgeEndPoint(lSite, bSite, vertex);
            //     }
            //     else {
            //         tArc->edge->setEdgeStartPoint(lSite, bSite, vertex);
            //     }
            //     this->beachline.rbInsertSuccessor(rArc->rbPrevious, newrArc);
            //     this->detachBeachSection(rArc);
            //     newrArc->edge = Edge::createEdge(newArc->site, rArc->site, nullptr, vertex);
            //     this->edges.push_back(newrArc->edge);
            //     this->cells[newArc->site->id]->halfEdges.push_back(HalfEdge::createHalfEdge(newrArc->edge, newArc->site, rArc->site));
            //     this->cells[rArc->site->id]->halfEdges.push_back(HalfEdge::createHalfEdge(newrArc->edge, rArc->site, newArc->site));
            // }
            
            Node* now = lArc->rbNext->rbNext;
            while(now != rArc && now != nullptr) {
                Node* tmp = now->rbNext;
                this->detachBeachSection(now);
                // this->beachline.rbRemoveNode(now);
                now = tmp;
            }

            this->attachCircleEvent(lArc);
            this->attachCircleEvent(rArc);
            // this->attachCircleEvent(newArc);

            return;
        }
    }

    void attachCircleEvent(Node* arc) {
        Node* lArc = arc->rbPrevious;
        Node* rArc = arc->rbNext;
        if (lArc == nullptr || rArc == nullptr) {
            return;
        }
        Site* lSite = lArc->site;
        Site* cSite = arc->site;
        Site* rSite = rArc->site;
        if (lSite == rSite) {
            return;
        }

        double bx = cSite->x;
        double by = cSite->y;
        double ax = lSite->x - bx;
        double ay = lSite->y - by;
        double cx = rSite->x - bx;
        double cy = rSite->y - by;
        double ar = lSite->r, br = cSite->r, cr = rSite->r;

        double d = 2 * (ax * cy - ay * cx);
        if (d >= -2e-3) return;
//        double ha = ax * ax + ay * ay;
//        double hc = cx * cx + cy * cy;
//        double x = (cy * ha - ay * hc) / d;
//        double y = (ax * hc - cx * ha) / d;
//        double ycenter = y + by;
//        Circle* circleEvent = new Circle();
//        circleEvent->arc = arc;
//        circleEvent->site = cSite;
//        circleEvent->x = x + bx;
//        circleEvent->y = ycenter + sqrt(x * x + y * y);
//        circleEvent->ycenter = ycenter;
//        arc->circleEvent = circleEvent;
//        std::cout << circleEvent->x << ' ' << circleEvent->y << ' ';
        double ha = ax * ax + ay * ay;
        double hc = cx * cx + cy * cy;
        double sqrt_ha = sqrt(ha);
        double sqrt_hc = sqrt(hc);
        double oa = (br * br - ar * ar + ha) / (2 * sqrt_ha);
        double oc = (br * br - cr * cr + hc) / (2 * sqrt_hc);
//        std::cout << "r " << ar << " " << br << ' ' << cr << ' ';
//        std::cout << "off " << oa << " " << oc << ' ';
        double la = - ax * (oa / sqrt_ha * ax) - ay * (oa / sqrt_ha * ay);
        double lc = - cx * (oc / sqrt_hc * cx) - cy * (oc / sqrt_hc * cy);
        double x = (ay * lc - cy * la) / (d/2);
        double y = (- ax * lc + cx * la) / (d/2);
//        std::cout << x + bx << ' ' << y + by + sqrt(x * x + y * y - br * br) << " circle" << std::endl;
        double xcenter = x + bx;
        double ycenter = y + by;
        Circle* circleEvent = new Circle();
        circleEvent->arc = arc;
        circleEvent->site = cSite;
        circleEvent->x = xcenter;
        circleEvent->y = ycenter + sqrt(x * x + y * y - br * br);
        circleEvent->ycenter = ycenter;
        if (circleEvent->y < this->sweepline){
            return;
        }
        arc->circleEvent = circleEvent;
//        std::cout << sqrt(x * x + y * y - br * br) << ' ' << sqrt((x-ax) * (x-ax) + (y-ay) * (y-ay) - ar * ar) << " " << sqrt((x-cx) * (x-cx) + (y-cy) * (y-cy) - cr * cr) << " circle" << std::endl;

        Circle* predecessor = nullptr;
        Circle* node = this->circleEvents.root;
        while (node) {
            if (circleEvent->y < node->y || (abs(circleEvent->y - node->y) < 1e-5 && circleEvent->x <= node->x)) {
                if (node->rbLeft) {
                    node = node->rbLeft;
                }
                else {
                    predecessor = node->rbPrevious;
                    break;
                }
            }
            else {
                if (node->rbRight) {
                    node = node->rbRight;
                }
                else {
                    predecessor = node;
                    break;
                }
            }
        }
        this->circleEvents.rbInsertSuccessor(predecessor, circleEvent);
        if (predecessor == nullptr) {
            this->firstCircleEvent = circleEvent;
        }
    }
    void detachCircleEvent(Node* arc) {
        Circle* circleEvent = arc->circleEvent;
        if (circleEvent) {
            if (circleEvent->rbPrevious == nullptr) {
                this->firstCircleEvent = circleEvent->rbNext;
            }
            if (!circleEvent)std::cout << "circle" << std::endl;
            this->circleEvents.rbRemoveNode(circleEvent);
            arc->circleEvent = nullptr;
        }
    }

    // cross product between (v1,v2) and (v3,v4)
//    double crossProduct(double v1x, double v1y, double v2x, double v2y, double v3x, double v3y, double v4x, double v4y) {
//        return (v2x - v1x) * (v4y - v3y) - (v2y - v1y) * (v4x - v3x);
//    }

    // get the intersection point of two edges
    Vertex* getIntersection(Edge* e1, Edge* e2){
        Vertex* v1 = e1->va;
        Vertex* v2 = e1->vb;
        Vertex* v3 = e2->va;
        Vertex* v4 = e2->vb;
        double x1 = v1->x, y1 = v1->y;
        double x2 = v2->x, y2 = v2->y;
        double x3 = v3->x, y3 = v3->y;
        double x4 = v4->x, y4 = v4->y;
        // judge if the two segments are intersected
        if (max(x1, x2) < min(x3, x4) || max(x3, x4) < min(x1, x2) || max(y1, y2) < min(y3, y4) || max(y3, y4) < min(y1, y2)) {
            return nullptr;
        }
        double d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
        if (abs(d) < 1e-3) {
            return nullptr;
        }
//        if (abs(crossProduct(x1, y1, x2, y2, x3, y3, x4, y4))<1e-3) {
//            return nullptr;
//        }
        // get the intersection point
        double x = (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4);
        double y = (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4);

        x /= d;
        y /= d;
        // judge if the intersection point is on the two segments
        if (x < min(x1, x2) || x > max(x1, x2) || x < min(x3, x4) || x > max(x3, x4)) {
            return nullptr;
        }
        if (y < min(y1, y2) || y > max(y1, y2) || y < min(y3, y4) || y > max(y3, y4)) {
            return nullptr;
        }
        return new Vertex(x, y);
    }

    void quantizeSites(std::vector<Site*>& sites) {
        double epsilon = this->epsilon;
        int n = sites.size();
        Site* site;
        while (n--) {
            site = sites[n];
            site->x = round(site->x / epsilon) * epsilon;
            site->y = round(site->y / epsilon) * epsilon;
        }
    }

    void computeByCGAL() {

    }

    void compute() {
        this->reset();

        std::vector<Site*> siteEvents = this->sites;
        // sort(siteEvents.begin(), siteEvents.end(), [](Site* a, Site* b) {
        //     return a->y < b->y || (a->y == b->y && a->x < b->x);
        //     });
        sort(siteEvents.begin(), siteEvents.end(), [](Site* a, Site* b) {
            double dy = (b->y) - (a->y);
            if (abs(dy) > 1e-3) return dy < 0;
            return (b->x - a->x) < 0;
        });

        Site* site;
        int i = siteEvents.size() - 1;
        site = siteEvents[i];
        int siteid = 0;
        double xsitex = -1;
        double xsitey = -1e9;
        Circle* circle;
        for (;;) {
            // std::cout << "xsitey " << xsitey << std::endl;
            circle = this->firstCircleEvent;
            if (site && (circle == nullptr || (site->y) < circle->y || (abs((site->y) - circle->y) < 1e-3 && site->x < circle->x))) {
                if (site->x != xsitex || site->y != xsitey) {
                    cells.push_back(Cell::createCell(site));
                    site->id = siteid++;
                    // std::cout<<"addBeachSection"<<std::endl;
                    this->addBeachSection(site);
                    xsitex = site->x;
                    xsitey = site->y;
                    // std::cout << "new site in " << site->id << " " << site->x << " " << site->y << std::endl;
                    this->sweepline = site->y;
                }
                if (i >= 1) site = siteEvents[--i];
                else site = nullptr;
            }
            else if (circle) {
                // std::cout<<"remove circle at " << circle->x << " " << circle->ycenter<< std::endl;
                // std::cout<<"circle's arc is at " << circle->arc->site->x << " " << circle->arc->site->y << std::endl;
                // std::cout<<"circle's arc's left is at " << circle->arc->rbPrevious->site->x << " " << circle->arc->rbPrevious->site->y << std::endl;
                // std::cout<<"circle's arc's right is at " << circle->arc->rbNext->site->x << " " << circle->arc->rbNext->site->y << std::endl;
                this->removeBeachSection(circle->arc);
                this->sweepline = circle->y;
            }
            else {
                break;
            }
        }

        this->clipEdges();
        this->closeCells();
        this->flipEdges();
    }

    void randomSites(int n) {
        std::default_random_engine generator;
        double width = this->bbox.xr - this->bbox.xl;
        double height = this->bbox.yb - this->bbox.yt;
        std::uniform_real_distribution<double> x_distribution(0, width);
        std::uniform_real_distribution<double> y_distribution(0, height);
        for (int i = 0; i < n; i++) {
            Site* site = new Site();
            site->x = x_distribution(generator);
            site->y = y_distribution(generator);
            this->sites.push_back(site);
        }
    }

    bool connectEdge(Edge* edge) {
        Vertex* vb = edge->vb;
        if (!!vb) {
            // std::cout << "vb " << vb->x << " " << vb->y << std::endl;
            return true;
        }
        Vertex* va = edge->va;
        if (va) {
            // std::cout << "va " << va->x << " " << va->y << std::endl;
        }
        double xl = this->bbox.xl;
        double xr = this->bbox.xr;
        double yt = this->bbox.yt;
        double yb = this->bbox.yb;
        Site* lSite = edge->lSite;
        Site* rSite = edge->rSite;
        double lx = lSite->x;
        double ly = lSite->y;
        double rx = rSite->x;
        double ry = rSite->y;

//        double fx = (lx + rx) / 2;
//        double fy = (ly + ry) / 2;

        double lr = lSite->r;
        double rr = rSite->r;
        double d2 = (lx-rx)*(lx-rx) + (ly-ry)*(ly-ry);
        double t = (lr*lr-rr*rr + d2) / (2*d2);
        double fx = lx * (1-t) + rx * (t);
        double fy = ly * (1-t) + ry * (t);
//        double fx = lx * (t) + rx * (1-t);
//        double fy = ly * (t) + ry * (1-t);
//        std::cout << lr <<  " " << rr << " " << t * sqrt(d2) - lr << " " << (1-t) * sqrt(d2) - rr << " check sector" << std::endl;
        // std::cout << lSite->id <<  " " << rSite->id << " check id" << std::endl;

        double fm;
        double fb;
        bool flag = false;
        this->cells[lSite->id]->closeMe = true;
        this->cells[rSite->id]->closeMe = true;
        if (abs(ry - ly) > 1e-5) {
            fm = (lx - rx) / (ry - ly);
            fb = fy - fm * fx;
            flag = true;
        }
        if (!flag) {
            if (fx<xl || fx>xr) return false;
            if (lx > rx) {
                if (!va || va->y < yt) va = Vertex::createVertex(fx, yt);
                else if (va->y >= yb) return false;
                vb = Vertex::createVertex(fx, yb);
            }
            else {
                if (!va || va->y > yb) {
                    va = Vertex::createVertex(fx, yb);
                }
                else if (va->y < yt)return false;
                vb = Vertex::createVertex(fx, yt);
            }
        }
        else if (fm < -1 || fm > 1) {
            if (lx > rx) {
                if (!va || va->y < yt)va = Vertex::createVertex((yt - fb) / fm, yt);
                else if (va->y >= yb)return false;
                vb = Vertex::createVertex((yb - fb) / fm, yb);
            }
            else {
                if (!va || va->y > yb)va = Vertex::createVertex((yb - fb) / fm, yb);
                else if (va->y < yt)return false;
                vb = Vertex::createVertex((yt - fb) / fm, yt);
            }
        }
        else {
            if (ly < ry) {
                if (!va || va->x < xl) {
                    va = Vertex::createVertex(xl, fm * xl + fb);
                }
                else if (va->x >= xr)return false;
                vb = Vertex::createVertex(xr, fm * xr + fb);
            }
            else {
                if (!va || va->x > xr) {
                    va = Vertex::createVertex(xr, fm * xr + fb);
                }
                else if (va->x < xl)return false;
                vb = Vertex::createVertex(xl, fm * xl + fb);
            }
        }
        edge->va = va;
        edge->vb = vb;
        return true;
    }

    bool clipEdge(Edge* edge) { // clip by boundary
        double ax = edge->va->x;
        double ay = edge->va->y;
        double bx = edge->vb->x;
        double by = edge->vb->y;
        double t0 = 0;
        double t1 = 1;
        double dx = bx - ax;
        double dy = by - ay;
        double q = ax - this->bbox.xl;
        if (abs(dx) < 1e-5 && q < 0)return false;
        double r = -q / dx;
        if (dx < 0) {
            if (r < t0)return false;
            if (r < t1)t1 = r;
        }
        else if (dx > 0) {
            if (r > t1)return false;
            if (r > t0)t0 = r;
        }
        q = this->bbox.xr - ax;
        if (abs(dx) < 1e-5 && q < 0)return false;
        r = q / dx;
        if (dx < 0) {
            if (r > t1)return false;
            if (r > t0)t0 = r;
        }
        else if (dx > 0) {
            if (r < t0)return false;
            if (r < t1)t1 = r;
        }
        q = ay - this->bbox.yt;
        if (abs(dy) < 1e-5 && q < 0)return false;
        r = -q / dy;
        if (dy < 0) {
            if (r < t0)return false;
            if (r < t1)t1 = r;
        }
        else if (dy > 0) {
            if (r > t1)return false;
            if (r > t0)t0 = r;
        }
        q = this->bbox.yb - ay;
        if (abs(dy) < 1e-5 && q < 0)return false;
        r = q / dy;
        if (dy < 0) {
            if (r > t1)return false;
            if (r > t0)t0 = r;
        }
        else if (dy > 0) {
            if (r < t0)return false;
            if (r < t1)t1 = r;
        }
        if (t0 > 0) {
            edge->va = Vertex::createVertex(ax + t0 * dx, ay + t0 * dy);
        }
        if (t1 < 1) {
            edge->vb = Vertex::createVertex(ax + t1 * dx, ay + t1 * dy);
        }
        if (t0 > 0 || t1 < 1) {
            this->cells[edge->lSite->id]->closeMe = true;
            this->cells[edge->rSite->id]->closeMe = true;
        }
        return true;
    }

    void clipEdges() {
        int iEdge = this->edges.size();
        Edge* edge;
        while (iEdge--) {
            edge = this->edges[iEdge];
            if (!this->connectEdge(edge) || !this->clipEdge(edge) ||
                (abs(edge->va->x - edge->vb->x) < 1e-6 && abs(edge->va->y - edge->vb->y) < 1e-6)) {
                if (edge->va && edge->vb)
                edge->va = nullptr;
                edge->vb = nullptr;
                this->edges.erase(this->edges.begin() + iEdge);
            }
        }
    }

    void closeCells() {
        double xl = this->bbox.xl;
        double xr = this->bbox.xr;
        double yt = this->bbox.yt;
        double yb = this->bbox.yb;
        int iCell = this->cells.size();
        Vertex* va, * vb, * vz;
        Cell* cell;
        Edge* edge;
        int nHalfEdges;
        int iLeft;
        bool lastBorderSegment;
        while (iCell--) {
            cell = this->cells[iCell];
            if (!cell->prepareHalfEdges()) {
                continue;
            }
            if (!cell->closeMe) {
                continue;
            }
            nHalfEdges = cell->halfEdges.size();
            iLeft = 0;
            while (iLeft < nHalfEdges) {
                va = cell->halfEdges[iLeft]->getEndPoint();
                vz = cell->halfEdges[(iLeft + 1) % nHalfEdges]->getStartPoint();
                if (abs(va->x - vz->x) > 1e-9 || abs(va->y - vz->y) > 1e-9) {
                    if (abs(va->x - xl) < this->epsilon && yb - va->y > this->epsilon) {
                        lastBorderSegment = abs(vz->x - xl) < this->epsilon;
                        vb = Vertex::createVertex(xl, lastBorderSegment ? vz->y : yb);
                        edge = Edge::createBorderEdge(cell->site, va, vb);
                        this->edges.push_back(edge);
                        iLeft++;
                        cell->halfEdges.insert(cell->halfEdges.begin() + iLeft, HalfEdge::createHalfEdge(edge, cell->site, nullptr));
                        nHalfEdges++;
                        if (lastBorderSegment) {
                            iLeft++;
                            continue;
                        }
                        va = vb;
                    }
                    if (abs(va->y - yb) < this->epsilon && xr - va->x > this->epsilon) {
                        lastBorderSegment = abs(vz->y - yb) < this->epsilon;
                        vb = Vertex::createVertex(lastBorderSegment ? vz->x : xr, yb);
                        edge = Edge::createBorderEdge(cell->site, va, vb);
                        this->edges.push_back(edge);
                        iLeft++;
                        cell->halfEdges.insert(cell->halfEdges.begin() + iLeft, HalfEdge::createHalfEdge(edge, cell->site, nullptr));
                        nHalfEdges++;
                        if (lastBorderSegment) {
                            iLeft++;
                            continue;
                        }
                        va = vb;
                    }
                    if (abs(va->x - xr) < this->epsilon && va->y - yt > this->epsilon) {
                        lastBorderSegment = abs(vz->x - xr) < this->epsilon;
                        vb = Vertex::createVertex(xr, lastBorderSegment ? vz->y : yt);
                        edge = Edge::createBorderEdge(cell->site, va, vb);
                        this->edges.push_back(edge);
                        iLeft++;
                        cell->halfEdges.insert(cell->halfEdges.begin() + iLeft, HalfEdge::createHalfEdge(edge, cell->site, nullptr));
                        nHalfEdges++;
                        if (lastBorderSegment) {
                            iLeft++;
                            continue;
                        }
                        va = vb;
                    }
                    if (abs(va->y - yt) < this->epsilon && va->x - xl > this->epsilon) {
                        lastBorderSegment = abs(vz->y - yt) < this->epsilon;
                        vb = Vertex::createVertex(lastBorderSegment ? vz->x : xl, yt);
                        edge = Edge::createBorderEdge(cell->site, va, vb);
                        this->edges.push_back(edge);
                        iLeft++;
                        cell->halfEdges.insert(cell->halfEdges.begin() + iLeft, HalfEdge::createHalfEdge(edge, cell->site, nullptr));
                        nHalfEdges++;
                        if (lastBorderSegment) {
                            iLeft++;
                            continue;
                        }
                        va = vb;

                        lastBorderSegment = abs(vz->x - xl) < this->epsilon;
                        vb = Vertex::createVertex(xl, lastBorderSegment ? vz->y : yb);
                        edge = Edge::createBorderEdge(cell->site, va, vb);
                        this->edges.push_back(edge);
                        iLeft++;
                        cell->halfEdges.insert(cell->halfEdges.begin() + iLeft, HalfEdge::createHalfEdge(edge, cell->site, nullptr));
                        nHalfEdges++;
                        if (lastBorderSegment) {
                            iLeft++;
                            continue;
                        }
                        va = vb;

                        lastBorderSegment = abs(vz->y - yb) < this->epsilon;
                        vb = Vertex::createVertex(lastBorderSegment ? vz->x : xr, yb);
                        edge = Edge::createBorderEdge(cell->site, va, vb);
                        this->edges.push_back(edge);
                        iLeft++;
                        cell->halfEdges.insert(cell->halfEdges.begin() + iLeft, HalfEdge::createHalfEdge(edge, cell->site, nullptr));
                        nHalfEdges++;
                        if (lastBorderSegment) {
                            iLeft++;
                            continue;
                        }
                        va = vb;

                        lastBorderSegment = abs(vz->x - xr) < this->epsilon;
                        vb = Vertex::createVertex(xr, lastBorderSegment ? vz->y : yt);
                        edge = Edge::createBorderEdge(cell->site, va, vb);
                        this->edges.push_back(edge);
                        iLeft++;
                        cell->halfEdges.insert(cell->halfEdges.begin() + iLeft, HalfEdge::createHalfEdge(edge, cell->site, nullptr));
                        nHalfEdges++;
                        if (lastBorderSegment) {
                            iLeft++;
                            continue;
                        }
                    }
                }
                iLeft++;
            }
            cell->closeMe = false;
        }
    }

    bool inside(std::vector<double> &p, std::tuple<double, double, double> &ln) {
        return std::get<0>(ln) * p[0] + std::get<1>(ln) * p[1] + std::get<2>(ln) < 0;
    }

    double dis_point2line(std::vector<double> &p, std::tuple<double, double, double> &ln) {
        return -(std::get<0>(ln) * p[0] + std::get<1>(ln) * p[1] + std::get<2>(ln)) / std::sqrt(pow(std::get<0>(ln), 2) + pow(std::get<1>(ln), 2));
    }

    std::vector<double> computeIntersection(std::vector<double> &v1, std::vector<double> &v2, std::tuple<double, double, double> &ln) {
        double a1 = std::get<0>(ln), b1 = std::get<1>(ln), c1 = std::get<2>(ln);
        double dx = v2[0] - v1[0], dy = v2[1] - v1[1];
        double a2 = -dy, b2 = dx, c2 = dy * v1[0] - dx * v1[1];
        double tmp = a1 * b2 - a2 * b1;
        double x = (c2 * b1 - c1 * b2) / tmp;
        double y = (a2 * c1 - a1 * c2) / tmp;
        std::vector<double> intersection;
        intersection.push_back(x);
        intersection.push_back(y);
        return intersection;
    }

    std::vector<double> computeIntersection(std::tuple<double, double, double> &ln1, std::tuple<double, double, double> &ln2) {
        double a1 = std::get<0>(ln1), b1 = std::get<1>(ln1), c1 = std::get<2>(ln1);
        double a2 = std::get<0>(ln2), b2 = std::get<1>(ln2), c2 = std::get<2>(ln2);
        double tmp = a1 * b2 - a2 * b1;
        double x = (c2 * b1 - c1 * b2) / tmp;
        double y = (a2 * c1 - a1 * c2) / tmp;
        std::vector<double> intersection;
        intersection.push_back(x);
        intersection.push_back(y);
        return intersection;
    }

    std::pair<std::vector<std::vector<double>>, std::vector<int>> makeCellBruteForce(std::vector<double> &v, std::vector<std::pair<int, std::tuple<double, double, double>>> &segments) {
        std::vector<std::vector<double>> outputList;
        if (this->convex_hull_boundary.size() > 0) {
            for (auto &boundary_vertex: this->convex_hull_boundary) {
                outputList.push_back(boundary_vertex);
            }
        } else {
//            const int boundary[4][2] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
            const double boundary[4][2] = {{this->bbox.xl, this->bbox.yt},
                                           {this->bbox.xr, this->bbox.yt},
                                           {this->bbox.xr, this->bbox.yb},
                                           {this->bbox.xl, this->bbox.yb}};
            for (int i = 0; i < 4; ++i) {
                std::vector<double> boundary_vertex;
                for (int j = 0; j < 2; ++j) {
                    boundary_vertex.push_back(boundary[i][j]);
                }
                outputList.push_back(boundary_vertex);
            }
        }

        std::vector<int> outputFlower = std::vector<int>(outputList.size(), -1);
        for (int i = 0; i < outputList.size(); ++i) {
            outputFlower[i] = -(i + 1);
        }
        std::vector<double> subjectCenter;
        subjectCenter.push_back(0.5 * RESOLUTION);
        subjectCenter.push_back(0.5 * RESOLUTION);
        double max_dist = 0.5 * RESOLUTION * std::sqrt(2);

        std::priority_queue<std::pair<double, int>> ln_order;
        for (int i = 0; i < segments.size(); ++i) {
            ln_order.push(std::make_pair(-dis_point2line(v, segments[i].second), i));
        }

        for (int i = 0; i < segments.size(); ++i) {
            int k = ln_order.top().second;
            ln_order.pop();
            int seg_id = segments[k].first;
            auto seg = segments[k].second;
            if (dis_point2line(subjectCenter, seg) > max_dist) {
                continue;
            }
            auto inputList(outputList);
            auto inputFlower(outputFlower);
            outputList.clear();
            outputFlower.clear();
            auto s = inputList.back();
            for (int kk = 0; kk < inputList.size(); ++kk) {
                auto e = inputList[kk];
                auto ef = inputFlower[kk];
                if (inside(e, seg)) {
                    if (!inside(s, seg)) {
                        outputList.push_back(computeIntersection(s, e, seg));
                        outputFlower.push_back(seg_id);
                    }
                    outputList.push_back(e);
                    outputFlower.push_back(ef);
                } else if (inside(s, seg)) {
                    outputList.push_back(computeIntersection(s, e, seg));
                    outputFlower.push_back(ef);
                }
                s = e;
            }
            if (outputList.size() < 3) {
                return std::make_pair(outputList, outputFlower);
            }
            subjectCenter[0] = subjectCenter[1] = 0;
            for (auto &p: outputList) {
                subjectCenter[0] += p[0];
                subjectCenter[1] += p[1];
            }
            subjectCenter[0] /= outputList.size();
            subjectCenter[1] /= outputList.size();

            max_dist = 0;
            for (auto &p: outputList) {
                max_dist = max(max_dist, distance(p, subjectCenter));
            }
        }
        for (auto &p: outputList) {
            p[0] /= RESOLUTION;
            p[1] /= RESOLUTION;
        }
        return std::make_pair(outputList, outputFlower);
    }

    std::pair<std::vector<std::vector<double>>, std::vector<int>> postClipByConvexHull(std::vector<std::vector<double>> cell, std::vector<int> flower, std::vector<std::tuple<double, double, double>> hull) {
        std::vector<std::vector<double>> outputList(cell);
        std::vector<int> outputFlower(flower);

        std::vector<double> subjectCenter;
        subjectCenter.push_back(0);
        subjectCenter.push_back(0);

        for (auto v: cell) {
            subjectCenter[0] += v[0];
            subjectCenter[1] += v[1];
        }
        subjectCenter[0] /= cell.size();
        subjectCenter[1] /= cell.size();

        double max_dist = 0;
        for (auto v: cell) {
            max_dist = max(max_dist, distance(subjectCenter, v));
        }

        for (auto seg: hull) {
            if (dis_point2line(subjectCenter, seg) > max_dist) {
                continue;
            }
            auto inputList(outputList);
            auto inputFlower(outputFlower);
            outputList.clear();
            outputFlower.clear();
            auto s = inputList.back();
            for (int kk = 0; kk < inputList.size(); ++kk) {
                auto e = inputList[kk];
                auto ef = inputFlower[kk];
                if (inside(e, seg)) {
                    if (!inside(s, seg)) {
                        outputList.push_back(computeIntersection(s, e, seg));
                        outputFlower.push_back(-1);
                    }
                    outputList.push_back(e);
                    outputFlower.push_back(ef);
                } else if (inside(s, seg)) {
                    outputList.push_back(computeIntersection(s, e, seg));
                    outputFlower.push_back(ef);
                }
                s = e;
            }
            if (outputList.size() < 3) {
                return std::make_pair(outputList, outputFlower);
            }
            subjectCenter[0] = subjectCenter[1] = 0;
            for (auto &p: outputList) {
                subjectCenter[0] += p[0];
                subjectCenter[1] += p[1];
            }
            subjectCenter[0] /= outputList.size();
            subjectCenter[1] /= outputList.size();

            max_dist = 0;
            for (auto &p: outputList) {
                max_dist = max(max_dist, distance(p, subjectCenter));
            }
        }
        return std::make_pair(outputList, outputFlower);
    }

    std::vector<std::pair<std::vector<std::vector<double>>, std::vector<int>>> computeBruteForce(bool clipHull=false) {
        // reset
        this->bbox = Bbox();
        this->bbox.xl = 0.0;
        this->bbox.xr = 1.0 * RESOLUTION;
        this->bbox.yt = 0.0;
        this->bbox.yb = 1.0 * RESOLUTION;
        int n = this->sites.size();
        std::vector<std::vector<std::tuple<double, double, double>>> lines;
        lines = std::vector<std::vector<std::tuple<double, double, double>>>(n, std::vector<std::tuple<double, double, double>>(n));
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                double d2 = pow(distance(this->sites[i], this->sites[j]), 2);
                double t = (this->sites[i]->w - this->sites[j]->w + d2) / (2 * d2);
                double x = this->sites[i]->x * (1 - t) + this->sites[j]->x * t;
                double y = this->sites[i]->y * (1 - t) + this->sites[j]->y * t;
                double a = this->sites[j]->x - this->sites[i]->x;
                double b = this->sites[j]->y - this->sites[i]->y;
                double c = - (a * x + b * y);
                lines[i][j] = std::make_tuple(a, b, c);
                lines[j][i] = std::make_tuple(-a, -b, -c);
            }
        }
        std::vector<std::pair<std::vector<std::vector<double>>, std::vector<int>>> ret;
        for (int i = 0; i < n; ++i) {
            std::vector<std::pair<int, std::tuple<double, double, double>>> segments;
            for (int j = 0; j < n; ++j) {
                if (i == j) continue;
                segments.push_back(std::make_pair(j, lines[i][j]));
            }
            std::vector<double> vertex;
            vertex.push_back(this->sites[i]->x);
            vertex.push_back(this->sites[i]->y);
            auto singleCF = makeCellBruteForce(vertex, segments);
            ret.push_back(singleCF);
        }

        if (!clipHull) {
            this->resultCellsFlowers = ret;
            return ret;
        }

        auto hull = findConvexHull(ret);
//        auto convex_hull_vertices = hull.first;
        auto convex_hull_edges = hull.second;

        for (int i = 0; i < n; ++i) {
//            std::cout << "bruteforce " << i << '\n';
//            if (i == 129) {
//                for (auto v: ret[i].first) {
//                    std::cout << v[0] << ' ' << v[1] << '\n';
//                }
//            }
            if (ret[i].first.size() == 0) {
                continue;
            }
            auto newCF = postClipByConvexHull(ret[i].first, ret[i].second, convex_hull_edges);
            ret[i] = newCF;
        }

        this->resultCellsFlowers = ret;
        return ret;
    }

    std::pair<std::vector<std::vector<double>>, std::vector<std::tuple<double, double, double>>> findConvexHull(std::vector<std::pair<std::vector<std::vector<double>>, std::vector<int>>> &noclipCFs) {
        std::vector<std::pair<double, int>> boundary_circles;
        int n = this->sites.size();
        for (int i = 0; i < n; ++i) {
            auto &singleCF = noclipCFs[i];
            auto cell = singleCF.first;
            auto flower = singleCF.second;
            double bend = -1;
            for (auto v: cell) {
                if (abs(v[1] - 0) < 1e-6) bend = v[0];
                else if (abs(v[0] - 1) < 1e-6) bend = v[1] + 1;
                else if (abs(v[1] - 1) < 1e-6) bend = 1 - v[0] + 2;
                else if (abs(v[0] - 0) < 1e-6) bend = 1 - v[1] + 3;
            }
            if (bend > -1) {
                boundary_circles.push_back(std::make_pair(bend, i));
            }
        }

        sort(boundary_circles.begin(), boundary_circles.end());
        // Find LTL
        int ltl = -1;
        for (int k = 0; k < boundary_circles.size(); ++k) {
            auto bc = boundary_circles[k];
            if (ltl == -1) {
                ltl = k;
            } else {
                Site *ltlsite = this->sites[boundary_circles[ltl].second], *bcsite = this->sites[bc.second];
                if ((ltlsite->y - ltlsite->r > bcsite->y - bcsite->r) ||
                    ((ltlsite->y - ltlsite->r == bcsite->y - bcsite->r) && (ltlsite->x - ltlsite->r > bcsite->x - bcsite->r))) {
                    ltl = k;
                }
            }
        }

        std::vector<int> convex_hull_circles;
        std::vector<std::tuple<double, double, double>> convex_hull_edges;

        if (boundary_circles.size() >= 2) {
            convex_hull_circles.push_back(boundary_circles[ltl].second);
            for (int k = ltl + 1; k <= ltl + boundary_circles.size(); ++k) {
                int cur_circle_id = boundary_circles[k % boundary_circles.size()].second;
                while (convex_hull_circles.size() > 0) {
                    std::tuple<double, double, double> tangency = rightSideCommonTangency(this->sites[convex_hull_circles.back()], this->sites[cur_circle_id]);
                    if (convex_hull_edges.size() == 0) {
                        convex_hull_circles.push_back(cur_circle_id);
                        convex_hull_edges.push_back(tangency);
                        break;
                    } else {
                        std::tuple<double, double, double> last_tengancy = convex_hull_edges.back();
                        if (std::get<0>(last_tengancy) * std::get<1>(tangency) - std::get<1>(last_tengancy) * std::get<0>(tangency) > 0) {
                            convex_hull_circles.push_back(cur_circle_id);
                            convex_hull_edges.push_back(tangency);
                            break;
                        } else {
                            convex_hull_circles.pop_back();
                            convex_hull_edges.pop_back();
                        }
                    }
                }
            }
        }

//        for (int cid: convex_hull_circles) std::cout << cid << " ";
//        std::cout << '\n';

        double center_x = 0.5 * RESOLUTION, center_y = 0.5 * RESOLUTION;

        std::vector<std::vector<double>> convex_hull_vertices;

        this->margin = 1.0 * RESOLUTION;

//        for (auto &edge: convex_hull_edges) {
//            double a = std::get<0>(edge), b = std::get<1>(edge), c = std::get<2>(edge);
//            std::cout << a <<  " " << b << " " << c << '\n';
//            double min_dist = abs(a * center_x + b * center_y + c) / sqrt(pow(a, 2) + pow(b, 2));
//            for (int i = 0; i < n; ++i) {
//                double dis = abs(a * this->sites[i]->x + b * this->sites[i]->y + c) / sqrt(pow(a, 2) + pow(b, 2));
//                std::cout << dis << '\n';
//                if (dis > 1e-6) {
//                    if (dis < min_dist) {
//                        min_dist = dis;
//                    }
//                }
//            }
//            if (this->margin > min_dist) {
//                this->margin = min_dist;
//            }
//        }

        for (auto &site: this->sites) {
            this->margin = min(site->r, this->margin);
        }

//        std::cout << "margin: " << this->margin << " " << this->margin / RESOLUTION << "\n";

        for (auto &edge: convex_hull_edges) {
            double dis = std::get<0>(edge) * center_x + std::get<1>(edge) * center_y + std::get<2>(edge);
            if (dis > 0) {
                std::get<2>(edge) += margin * sqrt(pow(std::get<0>(edge), 2) + pow(std::get<1>(edge), 2));
            } else {
                std::get<2>(edge) -= margin * sqrt(pow(std::get<0>(edge), 2) + pow(std::get<1>(edge), 2));
            }
        }

        this->convex_hull_boundary.clear();
        for (int k = 0; k < convex_hull_edges.size(); ++k) {
            auto v = computeIntersection(convex_hull_edges[k], convex_hull_edges[(k + 1) % convex_hull_edges.size()]);
            this->convex_hull_boundary.push_back(v);
            std::vector<double> tmp_v(v);
            tmp_v[0] /= RESOLUTION;
            tmp_v[1] /= RESOLUTION;
            convex_hull_vertices.push_back(tmp_v);
        }

//        this->convex_hull_boundary = convex_hull_vertices;

        this->convex_hull_boundary_edges.clear();
        // Rescale Tangency
        for (auto &edge: convex_hull_edges) {
            double a = std::get<0>(edge);
            double b = std::get<1>(edge);
            double c = std::get<2>(edge);
            this->convex_hull_boundary_edges.push_back(std::make_tuple(a, b, c));
            std::get<0>(edge) /= RESOLUTION;
            std::get<1>(edge) /= RESOLUTION;
            std::get<2>(edge) /= (RESOLUTION * RESOLUTION);
        }

        return std::make_pair(convex_hull_vertices, convex_hull_edges);
    }

    std::tuple<double, double, double> rightSideCommonTangency(Site* s1, Site* s2) {
        double d = distance(s1, s2);
        double l = s1->r - s2->r;
        double angle = asin(l / d);
        double theta = -(angle + PI / 2);
        double dx = s2->x - s1->x, dy = s2->y - s1->y;
        double nx = cos(theta) * dx - sin(theta) * dy, ny = sin(theta) * dx + cos(theta) * dy;
        double nl = sqrt(nx * nx + ny * ny);
        nx /= nl; ny /= nl;
        double s1tx = s1->x + nx * s1->r;
        double s1ty = s1->y + ny * s1->r;
        double s2tx = s2->x + nx * s2->r;
        double s2ty = s2->y + ny * s2->r;
        double b = -(s2tx - s1tx);
        double a = s2ty - s1ty;
        double c = - (a * s1tx + b * s1ty);
        return std::make_tuple(a, b, c);
    }

    std::vector<double> cellWeightedCenter(std::vector<Site*> sites) {
        double area = faceArea(sites);
//        std::cout << "area: " << area << std::endl;
        std::vector<double> circumcenter;
        circumcenter.push_back(sites[0]->x);
        circumcenter.push_back(sites[0]->y);
        std::vector<std::vector<double>> normals;
        for (int i = 0; i < 3; ++i) {
            std::vector<double> normal(2,0);
            double x = sites[(i + 1) % 3]->x - sites[i]->x;
            double y = sites[(i + 1) % 3]->y - sites[i]->y;
            normal[0] = -y;
            normal[1] = x;
            normals.push_back(normal);
        }
        double dx = 0, dy = 0;
        for (int i = 1; i < 3; ++i) {
            double nw = pow(distance(sites[0], sites[i]), 2) + sites[0]->w - sites[i]->w;
            auto normal = normals[(i + 1) % 3];
            dx += nw * normal[0];
            dy += nw * normal[1];
//            std::cout << "delta" <<i << "==="<< nw << " " << normal[0] << " " << normal[1]
//                << dx / (4 * area) << " " << dy / (4 * area) << std::endl;
        }
        circumcenter[0] += dx / (4 * area);
        circumcenter[1] += dy / (4 * area);
//        std::cout << "delta" << dx / (4 * area) << " " << dy / (4 * area) << std::endl;
        circumcenter[0] /= RESOLUTION;
        circumcenter[1] /= RESOLUTION;
        return circumcenter;
    }

    std::vector<std::vector<int>> relaxSites() {
        int iCell = this->cells.size();
        Cell* cell;
        Site* site;
        double dist;
        while (iCell--)
        {
            cell = cells[iCell];
            site = this->cellCentroid(cell);
            dist = this->distance(site, cell->site);
            // again = again || dist > 1e-3;
//            if (dist > 1e-2) {
                // if (dist > 1e4){
                //     std::cout<<"site id: "<<cell->site->id<<std::endl;
                //     std::cout<<"move site from"<<cell->site->x<<" "<<cell->site->y<<" to "<<site->x<<" "<<site->y<<std::endl;
                // }
                cell->site->x = site->x;
                cell->site->y = site->y;
                
//            }
            // if (rn > (1 - p)) {
            //     dist /= 2;
            //     Site* new_site = new Site(site->x + (site->x - cell->site->x) / dist, site->y + (site->y - cell->site->y) / dist);
            //     this->sites.push_back(new_site);
            // }
            // this->sites.push_back(site);
        }
        // this->compute();
        // if (again){
        //     this->relaxSites();
        // }
        std::vector<std::vector<int>> ret;
        for (auto site: this->sites) {
            std::vector<int> centroid;
            centroid.push_back(site->x);
            centroid.push_back(site->y);
            ret.push_back(centroid);
        }
        return ret;
    }

    void incrementRadius() {
        double k = 1e9;
        for (int i = 0; i < cells.size(); ++i) {
            Site *site = cells[i]->site;
            double r = cellInscribedCircleRadius(cells[i], site);
            k = min(k, r / site->r);
        }
        for (Site* site: this->sites) {
            site->r *= k;
            // std::cout << "increment to " << site->id << " " << site->r << std::endl;
        }
    }

    double distance(Site* a, Site* b) {
        return sqrt((a->x - b->x) * (a->x - b->x) + (a->y - b->y) * (a->y - b->y));
    }

    double distance(std::vector<double> &p, std::vector<double> &q) {
        return std::sqrt(pow(p[0] - q[0], 2) + pow(p[1] - q[1], 2));
    }

    double cellArea(Cell* cell) {
        double area = 0;
        int iHalfEdge = cell->halfEdges.size();
        HalfEdge* halfEdge;
        Vertex* p1, * p2;
        while (iHalfEdge--) {
            halfEdge = cell->halfEdges[iHalfEdge];
            p1 = halfEdge->getStartPoint();
            p2 = halfEdge->getEndPoint();
            area += (p1->x * p2->y - p2->x * p1->y);
        }
        area /= 2;
        return area;
    }
    double cellArea(std::vector<std::vector<double>> cell) {
        double area = 0;
        for (int i = 0; i < cell.size(); ++i) {
            auto p1 = cell[i];
            auto p2 = cell[(i + 1) % cell.size()];
            area += (p1[0] * p2[1] - p2[0] * p1[1]);
        }
        area /= 2;
        return abs(area);
    }
    double faceArea(std::vector<Site*> sites) {
//        return ((sites[1]->x - sites[0]->x) * (sites[2]->y - sites[0]->y) - (sites[2]->x - sites[0]->x) * (sites[1]->y - sites[0]->y)) / 2;
        double area = 0;
        for (int i = 0; i < sites.size(); ++i) {
            auto p1 = sites[i];
            auto p2 = sites[(i + 1) % sites.size()];
            area += (p1->x * p2->y - p2->x * p1->y);
        }
        area /= 2;
        return area;
    }
    Site* cellCentroid(Cell* cell) {
        double x = 0, y = 0;
        int iHalfEdge = cell->halfEdges.size();
        HalfEdge* halfEdge;
        Vertex* p1, * p2;
        double v = 0, area = 0;
        double min_len = 1e9;
        while (iHalfEdge--) {
            halfEdge = cell->halfEdges[iHalfEdge];
            p1 = halfEdge->getStartPoint();
            p2 = halfEdge->getEndPoint();
            // double dist = sqrt((p1->x-p2->x)*(p1->x-p2->x)+(p1->y-p2->y)*(p1->y-p2->y));
            // if (dist<1e3)continue;
            
            v = p1->x * p2->y - p2->x * p1->y;
            
            area += v;
            x += (p1->x + p2->x) * v;
            y += (p1->y + p2->y) * v;
            // TODO: precision error
        }
        // if (cell->site->id==312){
        //     std::cout<<"min_len: "<<min_len<<std::endl;
        // }
//        v = cellArea(cell) * 6;
//        return new Site(x / v, y / v);
// TODO precision error here
        area *= 3;
        return new Site(x / area, y / area);
    }
    std::vector<double> cellCentroid(std::vector<std::vector<double>> cell) {
        double x = 0, y = 0, area = 0;
        for (int i = 0; i < cell.size(); ++i) {
            auto p1 = cell[i];
            auto p2 = cell[(i + 1) % cell.size()];
            double v = p1[0] * p2[1] - p2[0] * p1[1];
            area += v;
            x += (p1[0] + p2[0]) * v;
            y += (p1[1] + p2[1]) * v;
        }
        area *= 3;
        std::vector<double> c;
        c.push_back(x / area);
        c.push_back(y / area);
        return c;
    }
    double cellInscribedCircleRadius(Cell* cell, Site* site) {
        double r = 1e9;
        int iHalfEdge = cell->halfEdges.size();
        HalfEdge* halfEdge;
        Vertex *p1, *p2;
        while (iHalfEdge--) {
            halfEdge = cell->halfEdges[iHalfEdge];
            p1 = halfEdge->getStartPoint();
            p2 = halfEdge->getEndPoint();
            double edgeLength = sqrt((p1->x - p2->x) * (p1->x - p2->x) + (p1->y - p2->y) * (p1->y - p2->y));
            double v = (p1->x - site->x) * (p2->y - site->y) - (p2->x - site->x) * (p1->y - site->y);
            r = min(r, abs(v / edgeLength));
        }
        return r;
    }

    // get the overlap of two cells and flip
    void flipEdges(){
        int iCell = this->cells.size();
        Cell* cell;
        Cell* cell1;
        while (iCell--)
        {
            cell = cells[iCell];
            int iHalfEdge = cell->halfEdges.size();
            HalfEdge* halfEdge;
            HalfEdge* halfEdge1;
            Edge* prevEdge = nullptr;
            Edge* nextEdge = nullptr;
            Edge* prevEdge1 = nullptr;
            Edge* nextEdge1 = nullptr;
            Site* site = cell->site;
            Site* site1 = nullptr;
            Vertex* v1, * v2 = nullptr;
            while (iHalfEdge--)
            {
                v1 = v2 = nullptr;
                halfEdge = cell->halfEdges[iHalfEdge];
                if (halfEdge->edge->rSite == nullptr)continue;
                // site1 = halfEdge->edge->rSite;
                if (halfEdge->edge->lSite == site){
                    site1 = halfEdge->edge->rSite;
                }
                else{
                    site1 = halfEdge->edge->lSite;
                }
                cell1 = this->cells[site1->id];
                prevEdge = cell->halfEdges[(iHalfEdge - 1 + cell->halfEdges.size()) % cell->halfEdges.size()]->edge;
                nextEdge = cell->halfEdges[(iHalfEdge + 1) % cell->halfEdges.size()]->edge;
                v1 = getIntersection(prevEdge, nextEdge);
                if (v1 == nullptr)continue;
                for (int i=0; i<cell1->halfEdges.size(); ++i){
                    halfEdge1 = cell1->halfEdges[i];
                    if (halfEdge1->edge->rSite == nullptr)continue;
                    if (halfEdge1->edge->rSite->id == site->id || halfEdge1->edge->lSite->id == site->id){
                        prevEdge1 = cell1->halfEdges[(i - 1 + cell1->halfEdges.size()) % cell1->halfEdges.size()]->edge;
                        nextEdge1 = cell1->halfEdges[(i + 1) % cell1->halfEdges.size()]->edge;
                        v2 = getIntersection(prevEdge1, nextEdge1);
                        break;
                    }
                }
                if (v2 == nullptr)continue;
                // // if v1 and v2 are not nullptr, it means that the two cells are overlapped
                // std::cout << "overlap: " << site->id << " " << site1->id << std::endl;
                // // flip the edge
                // std::cout<<"erase edge: "<<halfEdge->edge->va->x<<" "<<halfEdge->edge->va->y<<" "<<halfEdge->edge->vb->x<<" "<<halfEdge->edge->vb->y<<std::endl;
                // remove the halfEdge
                cell->halfEdges.erase(cell->halfEdges.begin() + iHalfEdge);
                // remove the halfEdge1
                cell1->halfEdges.erase(std::find(cell1->halfEdges.begin(), cell1->halfEdges.end(), halfEdge1));
                // remove the edge
                this->edges.erase(std::find(this->edges.begin(), this->edges.end(), halfEdge->edge));
                
                // replace prevEdge and nextEdge's vertex
                if (prevEdge->va == halfEdge->getStartPoint() || prevEdge->va == halfEdge->getEndPoint()){
                    prevEdge->va = v1;
                }
                else{
                    prevEdge->vb = v1;
                }
                if (nextEdge->va == halfEdge->getStartPoint() || nextEdge->va == halfEdge->getEndPoint()){
                    nextEdge->va = v1;
                }
                else{
                    nextEdge->vb = v1;
                }
                if (prevEdge1->va == halfEdge1->getStartPoint() || prevEdge1->va == halfEdge1->getEndPoint()){
                    prevEdge1->va = v2;
                }
                else{
                    prevEdge1->vb = v2;
                }
                if (nextEdge1->va == halfEdge1->getStartPoint() || nextEdge1->va == halfEdge1->getEndPoint()){
                    nextEdge1->va = v2;
                }
                else{
                    nextEdge1->vb = v2;
                }
                // add the new halfEdge
                Site* site2 = nullptr;
                if (prevEdge->lSite == site){
                    site2 = prevEdge->rSite;
                }
                else{
                    site2 = prevEdge->lSite;
                }
                Site* site3 = nullptr;
                if (nextEdge->lSite == site){
                    site3 = nextEdge->rSite;
                }
                else{
                    site3 = nextEdge->lSite;
                }
                Edge* newEdge = Edge::createEdge(site2, site3, v2, v1);
                // std::cout<<"new edge: "<<newEdge->va->x<<" "<<newEdge->va->y<<" "<<newEdge->vb->x<<" "<<newEdge->vb->y<<std::endl;
                HalfEdge* halfEdge2 = HalfEdge::createHalfEdge(newEdge, site2, site3);
                HalfEdge* halfEdge3 = HalfEdge::createHalfEdge(newEdge, site3, site2);
                Cell* cell2 = this->cells[site2->id];
                Cell* cell3 = this->cells[site3->id];
                cell2->halfEdges.push_back(halfEdge2);
                cell3->halfEdges.push_back(halfEdge3);
                this->edges.push_back(newEdge);
                cell2->prepareHalfEdges();
                cell3->prepareHalfEdges();
                break;
            }
        }
    }

    void inputSites(std::vector<std::pair<double, double>> positions) {
        for (auto p : positions) {
            this->sites.push_back(new Site(p.first * RESOLUTION, p.second * RESOLUTION));
        }
        return;
    }

    std::vector<std::tuple<double, double, double>> outputSites() {
        std::vector<std::tuple<double, double, double>> result;
        for (auto site : this->sites) {
            result.push_back(std::make_tuple(site->x / RESOLUTION, site->y / RESOLUTION, site->r / RESOLUTION));
        }
        return result;
    }

    void setRadius(std::vector<double> radius) {
        for (int i = 0; i < radius.size(); ++i) {
            sites[i]->r = radius[i] * RESOLUTION;
        }
    }

    void setWeight(std::vector<double> weight) {
        for (int i = 0; i < weight.size(); ++i) {
            sites[i]->w = weight[i] * pow(RESOLUTION, 2);
        }
    }

    void setBoundary(std::vector<std::vector<double>> vertices) {
        this->convex_hull_boundary.clear();
        for (auto v: vertices) {
            std::vector<double> pos;
            pos.push_back(v[0] * RESOLUTION);
            pos.push_back(v[1] * RESOLUTION);
            this->convex_hull_boundary.push_back(pos);
        }
        return;
    }

    void setCapacity(std::vector<double> cap) {
        this->capacity = cap;
    }

    std::vector<std::vector<std::pair<double, double>>> outputCells() {
        std::vector<std::vector<std::pair<double, double>>> result;
        for (auto site : this->sites) {
            std::vector<std::pair<double, double>> vertices;
            for (auto cell: cells) {
                if (cell->site == site) {
                    int iHalfEdge = cell->halfEdges.size();
                    HalfEdge* halfEdge;
                    Vertex* p;
                    while (iHalfEdge--) {
                        halfEdge = cell->halfEdges[iHalfEdge];
                        p = halfEdge->getStartPoint();
                        vertices.push_back(std::make_pair(p->x / RESOLUTION, p->y / RESOLUTION));
                    }
                    break;
                }
            }
            result.push_back(vertices);
        }
        return result;
    }

    std::vector<std::vector<double>> getVoronoi() {
        std::vector<std::vector<double>> result;
        for (auto edge: this->edges) {
            std::vector<double> e;
            if (edge->va) {
                e.push_back(edge->va->x / RESOLUTION);
                e.push_back(edge->va->y / RESOLUTION);
            }
            if (edge->vb) {
                e.push_back(edge->vb->x / RESOLUTION);
                e.push_back(edge->vb->y / RESOLUTION);
            }
            result.push_back(e);
        }
        return result;
    }

    std::vector<std::pair<int, int>> getNeighbors() {
        std::vector<std::pair<int, int>> result;
        std::vector<int> id_map(this->sites.size(), -1);
        for (int i = 0; i < this->sites.size(); ++i) {
            auto site = this->sites[i];
            id_map[site->id] = i;
        }
        for (auto edge: this->edges) {
//            if (edge->lSite) {
//                std::cout << "lSite " << edge->lSite->id << " ";
//            }
//            if (edge->rSite) {
//                std::cout << "rSite " << edge->rSite->id << " ";
//            }
//            std::cout << std::endl;
            if (edge->lSite && edge->rSite) {
                result.push_back(std::make_pair(id_map[edge->lSite->id], id_map[edge->rSite->id]));
            }
        }
        return result;
    }

    std::vector<std::vector<double>> getConvexHullVertices() {
        std::vector<std::vector<double>> ret;
        for (auto &v: this->convex_hull_boundary) {
            std::vector<double> tmp;
            tmp.push_back(v[0] / RESOLUTION);
            tmp.push_back(v[1] / RESOLUTION);
            ret.push_back(tmp);
        }
        return ret;
    }

    std::vector<std::tuple<double, double, double>> getConvexHullEdges() {
        std::vector<std::tuple<double, double, double>> ret;
        for (auto &edge: this->convex_hull_boundary_edges) {
            ret.push_back(std::make_tuple(
                std::get<0>(edge) / RESOLUTION,
                std::get<1>(edge) / RESOLUTION,
                std::get<2>(edge) / (RESOLUTION * RESOLUTION)
            ));
        }
        return ret;
    }

    std::vector<std::vector<double>> generateBoundarySites() {
        std::vector<std::vector<double>> ret;
        double d = this->margin;
        for (int i = 0; i < this->convex_hull_boundary.size(); ++i) {
            auto p1 = this->convex_hull_boundary[i], p2 = this->convex_hull_boundary[(i + 1) % this->convex_hull_boundary.size()];
            double l = distance(p1, p2);
            int num = ceil(l / d);
            for (int j = 0; j < num; ++j) {
                std::vector<double> dummy;
                dummy.push_back((p1[0] * (num - j) + p2[0] * j) / num / RESOLUTION);
                dummy.push_back((p1[1] * (num - j) + p2[1] * j) / num / RESOLUTION);
                ret.push_back(dummy);
            }
        }
        return ret;
    }

    std::vector<std::tuple<double, double, double>> getMaxInscribedCircles() {
        std::vector<std::tuple<double, double, double>> result;
        incrementRadius();
        for (auto site: sites) {
            for (int i = 0; i < cells.size(); ++i) {
                if (cells[i]->site == site) {
                    auto centroid = cellCentroid(cells[i]);
                    double r = cellInscribedCircleRadius(cells[i], centroid);
                    result.push_back(std::make_tuple(centroid->x / RESOLUTION, centroid->y / RESOLUTION, r / RESOLUTION));
                    break;
                }
            }
        }
        return result;
    }

    // Energy
    double inscribedCircleEnergy() {
        double value = 0;

        for (int k = 0; k < resultCellsFlowers.size(); ++k) {
            auto cell = resultCellsFlowers[k].first;
            auto site = cellCentroid(cell);
            double numerator = 0, denominator = 0;
            if (cell.size() == 0) {
                return 1e20;
            }
            for (int i = 0; i < cell.size(); ++i) {
                auto b = cell[i], a = cell[(i + 1) % cell.size()];
                double seg_len = distance(a, b);
                denominator += seg_len;
                double nx = a[1] - b[1], ny = -(a[0] - b[0]);
                nx /= seg_len;
                ny /= seg_len;
                double h = (a[0] - site[0]) * nx + (a[1] - site[1]) * ny;
                numerator += seg_len * h * h;
            }
            if (denominator == 0) value += 1;
            else {
                double Ac = PI * numerator / denominator, Ap = cellArea(cell);
                value += pow(Ac / Ap - 1, 2);
            }
        }

        return value;
    }

    double circumCircleEnergy() {
        double value = 0;

        for (int k = 0; k < resultCellsFlowers.size(); ++k) {
            auto cell = resultCellsFlowers[k].first;
            auto site = cellCentroid(cell);
            double numerator = 0, denominator = 0;
            if (cell.size() == 0) {
                return 1e20;
            }
            for (int i = 0; i < cell.size(); ++i) {
                auto b = cell[i], a = cell[(i + 1) % cell.size()];
                double seg_len = distance(a, b);
                denominator += seg_len;
                double dx = a[0] - b[0], dy = a[1] - b[1];
                double t2 = dx*dx+dy*dy,
                       t1 = 2 * ((b[0]-site[0])*dx + (b[1] - site[1])*dy),
                       t0 = pow(b[0]-site[0], 2) + pow(b[1]-site[1], 2);
                double tmp = (t2 / 3 + t1 / 2 + t0) * seg_len;
                numerator += tmp;
            }
            if (denominator == 0) value += 1;
            else {
                double Ac = PI * numerator / denominator, Ap = cellArea(cell);
                value += pow(Ac / Ap - 1, 2);
            }
        }

        return value;
    }

    double capacityEnergy() {
        double value = 0;

        for (int k = 0; k < resultCellsFlowers.size(); ++k) {
            auto cell = resultCellsFlowers[k].first;
            if (cell.size() == 0) {
                return 1e20;
            }
            double Ap = cellArea(cell);
//            value += pow(Ap / this->capacity[k] - 1, 2);
            value += pow(max(this->capacity[k] / Ap - 1, 0), 2);
//            value += pow(Ap / this->capacity[k] + this->capacity[k] / Ap, 2);
//            value += log(Ap / this->capacity[k] + this->capacity[k] / Ap);
//            value += log(max(this->capacity[k] / Ap, 1));
//            value += max(this->capacity[k] / Ap, 1);
        }

        return value;
    }

    double angleEnergy() {
        double value = 0;

        for (int k = 0; k < resultCellsFlowers.size(); ++k) {
            auto cell = resultCellsFlowers[k].first;
            auto flower = resultCellsFlowers[k].second;
            if (cell.size() == 0) {
                return 1e20;
            }
            int lenf = flower.size();
            double cos_reg = cos((lenf - 2) * PI / lenf);
            double tmp = 0;
            for (int f_id = 0; f_id < lenf; ++f_id) {
                auto va = cell[f_id], vb = cell[(f_id + 1) % lenf], vc = cell[(f_id + 2) % lenf];
                double a = distance(vb, vc), b = distance(vc, va), c = distance(va, vb);
                tmp += pow((b*b+c*c-a*a) / (2*b*c) - cos_reg, 2);
            }
            value += tmp / lenf;
        }

        return value;
    }

    double edgeEnergy() {
        double value = 0;

        for (int k = 0; k < resultCellsFlowers.size(); ++k) {
            auto cell = resultCellsFlowers[k].first;
            auto flower = resultCellsFlowers[k].second;
            if (cell.size() == 0) {
                return 1e20;
            }
            int lenf = flower.size();
            double tmp = 0;
            for (int f_id = 0; f_id < lenf; ++f_id) {
                auto va = cell[f_id], vb = cell[(f_id + 1) % lenf], vc = cell[(f_id + 2) % lenf];
                double a = distance(vb, vc), c = distance(va, vb);
                tmp += pow(a - c, 2);
            }
            value += tmp / lenf;
        }

        return value;
    }

    std::vector<double> gradientInterpolation(int N, std::vector<double> opt_var, std::vector<int> opt_indices, std::vector<double> energy_weights, double initial_delta) {
        std::vector<double> grad(3 * N, 0);
//        int (*energy_func)(void);
//        if (energy_type == 0) {
//           energy_func = &(this->inscribedCircleEnergy);
//        } else if (energy_type == 1) {
//           energy_func = &(this->circumCircleEnergy);
//        } else if (energy_type == 2) {
//           energy_func = &(this->capacityEnergy);
//        } else if (energy_type == 3) {
//           energy_func = &(this->angleEnergy);
//        } else if (energy_type == 4) {
//           energy_func = &(this->edgeEnergy);
//        }

        for (int k = 0; k < 2 * N; ++k) {
            opt_var[k] *= RESOLUTION;
        }

        for (int k = 2 * N; k < 3 * N; ++k) {
            opt_var[k] *= pow(RESOLUTION, 2);
        }

        for (int i: opt_indices) {
            double delta = initial_delta, deltaR;
            double *var;
            if (i < 2 * N) {
                int pos = i / 2, coord = i % 2;
                if (coord == 0) var = &(this->sites[pos]->x);
                else var = &(this->sites[pos]->y);
                deltaR = delta * RESOLUTION;
            } else {
                int pos = i - 2 * N;
                var = &(this->sites[pos]->w);
                deltaR = delta * pow(RESOLUTION, 2);
            }
//            std::cout << "begin grad\n";
            while (true) {
                *var = opt_var[i] + deltaR;

                auto res1 = computeBruteForce(true);
                double v1 = 0;
                if (energy_weights[0] > 0) {
                    v1 += energy_weights[0] * this->inscribedCircleEnergy();
                } else if (energy_weights[1] > 0) {
                    v1 += energy_weights[1] * this->circumCircleEnergy();
                } else if (energy_weights[2] > 0) {
                    v1 += energy_weights[2] * this->capacityEnergy();
                } else if (energy_weights[3] > 0) {
                    v1 += energy_weights[3] * this->angleEnergy();
                } else if (energy_weights[4] > 0) {
                    v1 += energy_weights[4] * this->edgeEnergy();
                }
//                std::cout << i << " v1 " << v1 << '\n';
                if (v1 > 1e9) {
                    delta /= 2;
                    deltaR /= 2;
                    continue;
                }

                *var = opt_var[i] - deltaR;
                auto res2 = computeBruteForce(true);
                double v2 = 0;
                if (energy_weights[0] > 0) {
                    v2 += energy_weights[0] * this->inscribedCircleEnergy();
                } else if (energy_weights[1] > 0) {
                    v2 += energy_weights[1] * this->circumCircleEnergy();
                } else if (energy_weights[2] > 0) {
                    v2 += energy_weights[2] * this->capacityEnergy();
                } else if (energy_weights[3] > 0) {
                    v2 += energy_weights[3] * this->angleEnergy();
                } else if (energy_weights[4] > 0) {
                    v2 += energy_weights[4] * this->edgeEnergy();
                }
//                std::cout << i << " v2 " << v2 << '\n';
                if (v2 > 1e9) {
                    delta /= 2;
                    deltaR /= 2;
                    continue;
                }

                *var = opt_var[i];
//                std::cout << i << ' ' << delta << '\n';
                grad[i] = (v1 - v2) / (2 * delta);
                break;
            }
        }
        return grad;
    }


};


PYBIND11_MODULE(Voronoi, m){
    m.doc() = "pybind11 example";
    pybind11::class_<Voronoi>(m, "Voronoi")
        .def( pybind11::init() )
        .def("reset", &Voronoi::reset)
        .def("clearSites", &Voronoi::clearSites)
        .def("randomSites", &Voronoi::randomSites)
        .def("compute", &Voronoi::compute)
        .def("computeBruteForce", &Voronoi::computeBruteForce)
        .def("computeByCGAL", &Voronoi::computeByCGAL)
        .def("relaxSites", &Voronoi::relaxSites)
        .def("incrementRadius", &Voronoi::incrementRadius)
        .def("setRadius", &Voronoi::setRadius)
        .def("setWeight", &Voronoi::setWeight)
        .def("setBoundary", &Voronoi::setBoundary)
        .def("inputSites", &Voronoi::inputSites)
        .def("getSites", &Voronoi::outputSites)
        .def("getCells", &Voronoi::outputCells)
        .def("getNeighbors", &Voronoi::getNeighbors)
        .def("getConvexHullVertices", &Voronoi::getConvexHullVertices)
        .def("getConvexHullEdges", &Voronoi::getConvexHullEdges)
        .def("generateBoundarySites", &Voronoi::generateBoundarySites)
        .def("getVoronoi", &Voronoi::getVoronoi)
        .def("getMaxInscribedCircles", &Voronoi::getMaxInscribedCircles)
        .def("inscribedCircleEnergy", &Voronoi::inscribedCircleEnergy)
        .def("circumCircleEnergy", &Voronoi::circumCircleEnergy)
        .def("capacityEnergy", &Voronoi::capacityEnergy)
        .def("angleEnergy", &Voronoi::angleEnergy)
        .def("edgeEnergy", &Voronoi::edgeEnergy)
        .def("setCapacity", &Voronoi::setCapacity)
        .def("gradientInterpolation", &Voronoi::gradientInterpolation);
}
