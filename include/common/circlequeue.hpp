/*
 * This file is part of GSPTucker.
 *
 * GSPTucker is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * GSPTucker is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GSPTucker.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef CIRCULAR_QUEUE_HPP_
#define CIRCULAR_QUEUE_HPP_

#include <condition_variable>
#include <mutex>
#include <stdexcept>

template <typename T>
class CircularQueue {
 private:
  int front, rear, size;
  T* queue;
  int capacity;

 public:
  CircularQueue(int capacity) : front(0), rear(0), size(0), capacity(capacity) {
    queue = new T[capacity];
  }

  ~CircularQueue() { delete[] queue; }

  CircularQueue(const CircularQueue&) = delete;
  CircularQueue& operator=(const CircularQueue&) = delete;

  void enqueue(const T& item) {
    queue[rear] = item;
    rear = (rear + 1) % capacity;
    size++;
  }

  bool dequeue(T& item) {
    item = queue[front];
    front = (front + 1) % capacity;
    size--;
    return true;
  }

  bool isEmpty() const { return size == 0; }

  bool isFull() const { return size == capacity; }

  int getSize() const { return size; }

  void resize(int newCapacity) {
    T* newQueue = new T[newCapacity];
    int i = 0;
    while (!isEmpty()) {
      dequeue(newQueue[i]);
      i++;
    }
    delete[] queue;
    queue = newQueue;
    front = 0;
    rear = i;
    capacity = newCapacity;
  }
};

#endif  // CIRCULAR_QUEUE_HPP_