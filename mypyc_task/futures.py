"""A Future class similar to the one in PEP 3148."""

__all__ = (
    'Future', 'wrap_future', 'isfuture',
)

import traceback
from collections.abc import Generator
from typing import Callable, Generic, TypeVar, Final, Literal
import concurrent.futures
import contextvars
import logging
import sys
import warnings

import asyncio
from asyncio import base_futures
from asyncio import events
from asyncio import exceptions
from asyncio import format_helpers


isfuture = base_futures.isfuture


_PENDING: Final[Literal["PENDING"]] = base_futures._PENDING
_CANCELLED: Final[Literal["CANCELLED"]] = base_futures._CANCELLED
_FINISHED: Final[Literal["FINISHED"]] = base_futures._FINISHED


STACK_DEBUG = logging.DEBUG - 1  # heavy-duty debugging


_T = TypeVar("_T")

class Future(Generic[_T]):
    """This class is *almost* compatible with concurrent.futures.Future.

    Differences:

    - This class is not thread-safe.

    - result() and exception() do not take a timeout argument and
      raise an exception when the future isn't done yet.

    - Callbacks registered with add_done_callback() are always called
      via the event loop's call_soon().

    - This class is not compatible with the wait() and as_completed()
      methods in the concurrent.futures package.

    (In Python 3.4 or later we may be able to unify the implementations.)
    """

    # Class variables serving as defaults for instance variables.
    _state: Literal["PENDING", "CANCELLED", "FINISHED"] = _PENDING
    _result: _T | None = None
    _exception: BaseException | None = None
    _loop: asyncio.AbstractEventLoop | None = None
    _source_traceback: traceback.StackSummary | None = None
    _cancel_message: str | None = None
    # A saved CancelledError for later chaining as an exception context.
    _cancelled_exc: asyncio.CancelledError | None = None

    # This field is used for a dual purpose:
    # - Its presence is a marker to declare that a class implements
    #   the Future protocol (i.e. is intended to be duck-type compatible).
    #   The value must also be not-None, to enable a subclass to declare
    #   that it is not compatible by setting this to None.
    # - It is set by __iter__() below so that Task._step() can tell
    #   the difference between
    #   `await Future()` or`yield from Future()` (correct) vs.
    #   `yield Future()` (incorrect).
    _asyncio_future_blocking: bool = False

    __log_traceback: bool = False

    def __init__(self, *, loop: asyncio.AbstractEventLoop | None=None):
        """Initialize the future.

        The optional event_loop argument allows explicitly setting the event
        loop object used by the future. If it's not provided, the future uses
        the default event loop.
        """
        self._source_traceback: traceback.StackSummary | None = None
        if loop is None:
            self._loop = events._get_event_loop()  # type: ignore[attr-defined]
        else:
            self._loop = loop
        assert self._loop is not None
        self._callbacks: "list[tuple[Callable[[Future[_T]], object], contextvars.Context]]" = []
        if self._loop.get_debug():
            self._source_traceback = format_helpers.extract_stack(
                sys._getframe(1))

    def __repr__(self) -> str:
        return base_futures._future_repr(self)  # type: ignore[attr-defined, no-any-return]

    def __del__(self) -> None:
        assert self._loop is not None
        if not self.__log_traceback:
            # set_exception() was not called, or result() or exception()
            # has consumed the exception
            return
        exc = self._exception
        context: dict[str, BaseException | traceback.StackSummary | Future[_T] | str | None] = {
            'message':
                f'{self.__class__.__name__} exception was never retrieved',
            'exception': exc,
            'future': self,
        }
        if self._source_traceback:
            context['source_traceback'] = self._source_traceback
        self._loop.call_exception_handler(context)

    @property
    def _log_traceback(self) -> bool:
        return self.__log_traceback

    @_log_traceback.setter
    def _log_traceback(self, val: object) -> None:
        if val:
            raise ValueError('_log_traceback can only be set to False')
        self.__log_traceback = False

    def get_loop(self) -> asyncio.AbstractEventLoop:
        """Return the event loop the Future is bound to."""
        loop = self._loop
        if loop is None:
            raise RuntimeError("Future object is not initialized.")
        return loop

    def _make_cancelled_error(self) -> asyncio.CancelledError:
        """Create the CancelledError to raise if the Future is cancelled.

        This should only be called once when handling a cancellation since
        it erases the saved context exception value.
        """
        if self._cancelled_exc is not None:
            exc = self._cancelled_exc
            self._cancelled_exc = None
            return exc

        if self._cancel_message is None:
            exc = exceptions.CancelledError()
        else:
            exc = exceptions.CancelledError(self._cancel_message)
        exc.__context__ = self._cancelled_exc
        # Remove the reference since we don't need this anymore.
        self._cancelled_exc = None
        return exc

    def cancel(self, msg: str | None=None) -> bool:
        """Cancel the future and schedule callbacks.

        If the future is already done or cancelled, return False.  Otherwise,
        change the future's state to cancelled, schedule the callbacks and
        return True.
        """
        if msg is not None:
            warnings.warn("Passing 'msg' argument to Future.cancel() "
                          "is deprecated since Python 3.11, and "
                          "scheduled for removal in Python 3.14.",
                          DeprecationWarning, stacklevel=2)
        self.__log_traceback = False
        if self._state != _PENDING:
            return False
        self._state = _CANCELLED
        self._cancel_message = msg
        self.__schedule_callbacks()
        return True

    def __schedule_callbacks(self) -> None:
        """Internal: Ask the event loop to call all callbacks.

        The callbacks are scheduled to be called as soon as possible. Also
        clears the callback list.
        """
        assert self._loop is not None
        callbacks = self._callbacks[:]
        if not callbacks:
            return

        self._callbacks[:] = []
        for callback, ctx in callbacks:
            self._loop.call_soon(callback, self, context=ctx)

    def cancelled(self) -> bool:
        """Return True if the future was cancelled."""
        return self._state == _CANCELLED

    # Don't implement running(); see http://bugs.python.org/issue18699

    def done(self) -> bool:
        """Return True if the future is done.

        Done means either that a result / exception are available, or that the
        future was cancelled.
        """
        return self._state != _PENDING

    def result(self) -> _T:
        """Return the result this future represents.

        If the future has been cancelled, raises CancelledError.  If the
        future's result isn't yet available, raises InvalidStateError.  If
        the future is done and has an exception set, this exception is raised.
        """
        if self._state == _CANCELLED:
            exc = self._make_cancelled_error()
            raise exc
        if self._state != _FINISHED:
            raise exceptions.InvalidStateError('Result is not ready.')
        self.__log_traceback = False
        if self._exception is not None:
            raise self._exception.with_traceback(self._exception_tb)
        return self._result  # type: ignore[return-value]

    def exception(self) -> BaseException | None:
        """Return the exception that was set on this future.

        The exception (or None if no exception was set) is returned only if
        the future is done.  If the future has been cancelled, raises
        CancelledError.  If the future isn't done yet, raises
        InvalidStateError.
        """
        if self._state == _CANCELLED:
            exc = self._make_cancelled_error()
            raise exc
        if self._state != _FINISHED:
            raise exceptions.InvalidStateError('Exception is not set.')
        self.__log_traceback = False
        return self._exception

    def add_done_callback(self, fn: Callable[[Future[_T]], object], *, context: contextvars.Context | None=None) -> None:
        """Add a callback to be run when the future becomes done.

        The callback is called with a single argument - the future object. If
        the future is already done when this is called, the callback is
        scheduled with call_soon.
        """
        assert self._loop is not None
        if self._state != _PENDING:
            self._loop.call_soon(fn, self, context=context)
        else:
            if context is None:
                context = contextvars.copy_context()
            self._callbacks.append((fn, context))

    # New method not in PEP 3148.

    def remove_done_callback(self, fn: Callable[[Future[_T]], object]) -> int:
        """Remove all instances of a callback from the "call when done" list.

        Returns the number of callbacks removed.
        """
        filtered_callbacks = [(f, ctx)
                              for (f, ctx) in self._callbacks
                              if f != fn]
        removed_count = len(self._callbacks) - len(filtered_callbacks)
        if removed_count:
            self._callbacks[:] = filtered_callbacks
        return removed_count

    # So-called internal methods (note: no set_running_or_notify_cancel()).

    def set_result(self, result: _T) -> None:
        """Mark the future done and set its result.

        If the future is already done when this method is called, raises
        InvalidStateError.
        """
        if self._state != _PENDING:
            raise exceptions.InvalidStateError(f'{self._state}: {self!r}')
        self._result = result
        self._state = _FINISHED
        self.__schedule_callbacks()

    def set_exception(self, exception: BaseException) -> None:
        """Mark the future done and set an exception.

        If the future is already done when this method is called, raises
        InvalidStateError.
        """
        if self._state != _PENDING:
            raise exceptions.InvalidStateError(f'{self._state}: {self!r}')
        if isinstance(exception, type):
            exception = exception()
        if type(exception) is StopIteration:
            raise TypeError("StopIteration interacts badly with generators "
                            "and cannot be raised into a Future")
        self._exception = exception
        self._exception_tb = exception.__traceback__
        self._state = _FINISHED
        self.__schedule_callbacks()
        self.__log_traceback = True

    def __await__(self) -> Generator[Future[_T], object, _T]:
        if not self.done():
            self._asyncio_future_blocking = True
            yield self  # This tells Task to wait for completion.
        if not self.done():
            raise RuntimeError("await wasn't used with future")
        return self.result()    # May raise too.

    __iter__ = __await__  # make compatible with 'yield from'.


# Needed for testing purposes.
_PyFuture: Final = Future

from asyncio.futures import _get_loop as _get_loop  # type: ignore[attr-defined]
from asyncio.futures import _chain_future  # type: ignore[attr-defined]
from asyncio.futures import wrap_future

_CFuture: Final = Future
