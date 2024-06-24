# Guía de instalación de kokkos

<p align=center>
<img src="imagenes/kokkos-logo.png">
</P>

## Descarga y construye el paquete de Kokkos

```bash
    $ git clone https://github.com/kokkos/kokkos Kokkos_build
    $ cd Kokkos_build
    $ nano CMakeLists.txt
```

Ubica las lineas y agrega:

```
    ...
    SET(KOKKOS_GIVEN_VARIABLES)
    FOREACH (var ${_variableNames})
    ...
```

Agrega las siguientes lineas para establecer Cuda y OpenMP como dispositivos

```
    ...
    SET(KOKKOS_GIVEN_VARIABLES)
    SET(Kokkos_ENABLE_CUDA DEFAULT CACHE BOOL "")
    SET(Kokkos_ENABLE_OPENMP DEFAULT CACHE BOOL "")
    FOREACH (var ${_variableNames})
    ...
```

Ahora construiremos Kokkos con OpenMP, Cuda y serial

**Serial**

```bash
    $ mkdir build_serial && cd build_serial
    $ cmake -DCMAKE_INSTALL_PREFIX=${HOME}/Kokkos_Devices/Kokkos_Serial \ -DKokkos_ENABLE_SERIAL=Off ..
    $ make -j 8
    $ make install
    $ cd ..
```

**OpenMP**

```bash
    $ mkdir build_openmp && cd build_openmp
    $ cmake -DCMAKE_INSTALL_PREFIX=${HOME}/Kokkos_Devices/Kokkos_OpenMP \ -DKokkos_ENABLE_OPENMP=On \ -DKokkos_ENABLE_SERIAL=Off ..
    $ make -j 8
    $ make install
    $ cd ..
```

**Cuda**

```bash
    $ mkdir build_cuda && cd build_cuda
    $ cmake -DCMAKE_INSTALL_PREFIX=${HOME}/Kokkos_Devices/Kokkos_Cuda \ -DKokkos_ENABLE_CUDA=On \ -DKokkos_ENABLE_SERIAL=Off ..
    $ make -j 8
    $ make install
    $ cd ..
```
Vamos a crear un archivo básico para analizar el comportamiento de kokkos en los distintos dispositivos, para esto:

````bash
    $ mkdir test
    $ nano main.cpp
````
Copia el siguiente código y guardalo
````cpp
#include <Kokkos_Core.hpp>
#include <iostream>
#include <chrono>

// Kernel para la multiplicación de matrices
void matrixMultiply(const int n) {
    // Definir las matrices
    Kokkos::View<double**> A("A", n, n);
    Kokkos::View<double**> B("B", n, n);
    Kokkos::View<double**> C("C", n, n);

    // Inicializar las matrices A y B
    Kokkos::parallel_for("InitializeA", n, KOKKOS_LAMBDA(const int i) {
        for (int j = 0; j < n; ++j) {
            A(i,j) = static_cast<double>(i + j);
            B(i,j) = static_cast<double>(i - j);
        }
    });

    // Realizar la multiplicación de matrices
    auto start = std::chrono::high_resolution_clock::now();
    Kokkos::parallel_for("MatrixMultiply", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {n,n}),
                         KOKKOS_LAMBDA(const int i, const int j) {
        double sum = 0.0;
        for (int k = 0; k < n; ++k) {
            sum += A(i,k) * B(k,j);
        }
        C(i,j) = sum;
    });
    Kokkos::fence();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    // Imprimir el tiempo de ejecución
    std::cout << "Tiempo de ejecución: " << duration.count() << " segundos\n";
}

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);

    #ifdef KOKKOS_ENABLE_CUDA
    printf("Corriendo on GPU\n");
    #elif defined(KOKKOS_ENABLE_OPENMP)
    printf("Corriendo on CPU - OPENMP\n");
    #else
    printf("Corriendo on CPU - SERIAL\n");
    #endif

    {
        const int n = 1000; // Tamaño de las matrices
        matrixMultiply(n);
    }

    Kokkos::finalize();
    return 0;
}
````
Ahora compilaremos el archivo para los distintos dispositivos y revisaremos la salida (principalmente del dispositivo en el que está corriendo)

Crea una carpeta para ahí configurar la compilación

```bash
    $ mkdir build
```


**SERIAL**

Entrada
````bash
    $ export Kokkos_DIR=${HOME}/Kokkos_Devices/Kokkos_Serial
    $ cd ./build
    $ rm CMakeCache.txt
    $ make clean
    $ cmake ..
    $ make
    $ ./main
````
Salida
````
````

**OPENMP**

Entrada
````bash
    $ export Kokkos_DIR=${HOME}/Kokkos_Devices/Kokkos_OpenMP
    $ cd ./build
    $ rm CMakeCache.txt
    $ make clean
    $ cmake ..
    $ make
    $ ./main
````
Salida
````
````

**CUDA**

Entrada
````bash
    $ export Kokkos_DIR=${HOME}/Kokkos_Devices/Kokkos_Cuda
    $ cd ./build
    $ rm CMakeCache.txt
    $ make clean
    $ cmake ..
    $ make
    $ ./main
````
Salida
````
````

De esta manera configuraremos kokkos para correr un mismo código en diferentes dispositivos :)