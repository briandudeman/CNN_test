#ifndef LAYER
#define LAYER

class Layer{
public:


    virtual void forward() = 0;

    virtual void backward() = 0;

protected:
    int lr = 0.0001;
};

#endif