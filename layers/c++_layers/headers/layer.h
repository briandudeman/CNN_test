#ifndef LAYER
#define LAYER

class Layer{
public:

    Layer();

    virtual void forward() = 0;

    virtual void backward() = 0;

protected:
    int lr = 0.0001;
};

#endif