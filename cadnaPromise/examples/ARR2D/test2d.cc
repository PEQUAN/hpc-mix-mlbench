#include <iostream>

int main(){
    
    __PROMISE__ h = 3.4;
    float arr[3][4] = {{1.2, 2.4, 2.3, 2}, {1.2, 1.4, 2.3, 2 }, {9.2, 5.4, 5.3, 2}};

    float **clusters;
    clusters = (float **)malloc(3 * sizeof(float *));
    
    for (int i=0; i < 3; i++){
        clusters[i] = (float *)malloc(4 * sizeof(float));
        for (int j=0; j < 4; j++){
            
            clusters[i][j] = arr[i][j];
        }
    }
        
    
    for (int i=0; i < 3; i++){
        PROMISE_CHECK_ARRAY(clusters[i], 4);
    }
    
    PROMISE_CHECK_ARRAY2D(clusters, 3, 4);
   
    
    return 0;
}

