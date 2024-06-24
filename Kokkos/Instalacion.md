# Guía de instalación de kokkos

<p align=center>
<img src="imagenes/kokkos-logo.png">
</P>

## Descarga y construye el paquete de Kokkos

```bash
    $ git clone https://github.com/kokkos/kokkos Kokkos_build
    $ cd Kokkos_build
```

Ahora construiremos Kokkos con OpenMP, Cuda y serial

**Serial**

```bash
    $ mkdir build_serial && cd build_serial
    $ cmake -DCMAKE_INSTALL_PREFIX=${HOME}/Kokkos_Devices/Kokkos_Serial \ -DKokkos_ENABLE_SERIAL=On ..
    $ make -j 8
    $ make install
    $ cd ..
```

**OpenMP**

Edita el archivo **CMakeLists.txt** y ubica las lineas:

```
    ...
    SET(KOKKOS_GIVEN_VARIABLES)
    FOREACH (var ${_variableNames})
    ...
```

Agrega las siguientes lineas para establecer OpenMP como dispositivos

```
    ...
    SET(KOKKOS_GIVEN_VARIABLES)
    SET(Kokkos_ENABLE_OPENMP DEFAULT CACHE BOOL "")
    FOREACH (var ${_variableNames})
    ...
```

Despues de guardar el archivo CMake:

```bash
    $ mkdir build_openmp && cd build_openmp
    $ cmake -DCMAKE_INSTALL_PREFIX=${HOME}/Kokkos_Devices/Kokkos_OpenMP \ -DKokkos_ENABLE_OPENMP=On ..
    $ make -j 8
    $ make install
    $ cd ..
```

**Cuda**

Agrega las siguiente linea en el archivo **CMakeLists.txt** para establecer Cuda como dispositivo

```
    ...
    SET(KOKKOS_GIVEN_VARIABLES)
    SET(Kokkos_ENABLE_CUDA DEFAULT CACHE BOOL "")
    FOREACH (var ${_variableNames})
    ...
```
Despues de guardar el archivo CMake:

```bash
    $ mkdir build_cuda && cd build_cuda
    $ cmake -DCMAKE_INSTALL_PREFIX=${HOME}/Kokkos_Devices/Kokkos_Cuda -DKokkos_ENABLE_CUDA=On ..
    $ make -j 8
    $ make install
    $ cd ..
```
Vamos a crear un archivo básico para analizar el comportamiento de kokkos en los distintos dispositivos, para esto:

````bash
    $ mkdir test
    $ cd test
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
    printf("Corriendo en GPU\n");
    #elif defined(KOKKOS_ENABLE_OPENMP)
    printf("Corriendo en CPU - OPENMP\n");
    #else
    printf("Corriendo en CPU - SERIAL\n");
    #endif

    {
        const int n = 1000; // Tamaño de las matrices
        matrixMultiply(n);
    }

    Kokkos::finalize();
    return 0;
}
````

Crea el archivo CMakeLists.txt y guardalo

````bash
    $ nano CMakeLists.txt
````
````CMake
cmake_minimum_required (VERSION 3.10)
project (main)

find_package(Kokkos REQUIRED)

add_executable(main main.cpp)
target_link_libraries(main Kokkos::kokkos)
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
    $ cd build
    $ cmake ..
    $ make
    $ ./main
    $ cd ..
````
Salida
````
Corriendo en CPU - SERIAL
Tiempo de ejecución: ### segundos
````

**OPENMP**

Entrada
````bash
    $ export Kokkos_DIR=${HOME}/Kokkos_Devices/Kokkos_OpenMP
    $ cd build
    $ rm CMakeCache.txt
    $ make clean
    $ cmake ..
    $ make
    $ ./main
    $ cd ..
````
Salida
````
Corriendo en CPU - OPENMP
Tiempo de ejecución: ### segundos
````

**CUDA**

Entrada
````bash
    $ export Kokkos_DIR=${HOME}/Kokkos_Devices/Kokkos_Cuda
    $ cd build
    $ rm CMakeCache.txt
    $ make clean
    $ cmake ..
    $ make
    $ ./main
````
Salida
````
Corriendo en GPU
Tiempo de ejecución: ### segundos
````

De esta manera configuraremos kokkos para correr un mismo código en diferentes dispositivos :)