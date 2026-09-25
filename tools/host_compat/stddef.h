#ifndef HOST_COMPAT_STDDEF_H
#define HOST_COMPAT_STDDEF_H

#ifdef __cplusplus
extern "C" {
#endif

typedef unsigned __int64 size_t;
typedef __int64 ptrdiff_t;
typedef __int64 intptr_t;
typedef unsigned __int64 uintptr_t;

#ifndef NULL
#ifdef __cplusplus
#define NULL 0
#else
#define NULL ((void *)0)
#endif
#endif

#ifdef __cplusplus
}
#endif

#endif // HOST_COMPAT_STDDEF_H
