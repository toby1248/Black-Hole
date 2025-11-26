/**
 * Minimal OrbitControls for non-module Three.js builds.
 * Supports rotate (LMB), dolly (wheel), and pan (RMB / Shift+LMB).
 * Adapted from Three.js examples (MIT License).
 */
(function () {
    if (typeof THREE === 'undefined' || THREE.OrbitControls) {
        return;
    }

    const STATE = { NONE: -1, ROTATE: 0, DOLLY: 1, PAN: 2 };

    THREE.OrbitControls = function (object, domElement) {
        this.object = object;
        this.domElement = domElement || document;

        // parameters
        this.enabled = true;
        this.enableDamping = false;
        this.dampingFactor = 0.05;
        this.enablePan = true;
        this.enableZoom = true;
        this.enableRotate = true;
        this.minDistance = 0;
        this.maxDistance = Infinity;

        // internals
        const scope = this;
        const target = new THREE.Vector3();
        const spherical = new THREE.Spherical();
        const sphericalDelta = new THREE.Spherical();
        const panOffset = new THREE.Vector3();
        let scale = 1;
        let state = STATE.NONE;
        let rotateStart = new THREE.Vector2();
        let rotateEnd = new THREE.Vector2();
        let rotateDelta = new THREE.Vector2();
        let panStart = new THREE.Vector2();
        let panEnd = new THREE.Vector2();
        let panDelta = new THREE.Vector2();
        let dollyStart = new THREE.Vector2();
        let dollyEnd = new THREE.Vector2();
        let dollyDelta = new THREE.Vector2();

        this.update = function () {
            const offset = new THREE.Vector3();
            offset.copy(scope.object.position).sub(target);
            spherical.setFromVector3(offset);

            spherical.theta += sphericalDelta.theta;
            spherical.phi += sphericalDelta.phi;
            spherical.phi = Math.max(0.000001, Math.min(Math.PI - 0.000001, spherical.phi));
            spherical.makeSafe();

            spherical.radius *= scale;
            spherical.radius = Math.max(scope.minDistance, Math.min(scope.maxDistance, spherical.radius));

            target.add(panOffset);
            offset.setFromSpherical(spherical);
            scope.object.position.copy(target).add(offset);
            scope.object.lookAt(target);

            if (scope.enableDamping) {
                sphericalDelta.theta *= (1 - scope.dampingFactor);
                sphericalDelta.phi *= (1 - scope.dampingFactor);
                panOffset.multiplyScalar(1 - scope.dampingFactor);
            } else {
                sphericalDelta.set(0, 0, 0);
                panOffset.set(0, 0, 0);
            }
            scale = 1;
            return true;
        };

        function handleMouseDownRotate(event) {
            rotateStart.set(event.clientX, event.clientY);
            state = STATE.ROTATE;
        }

        function handleMouseDownDolly(event) {
            dollyStart.set(event.clientX, event.clientY);
            state = STATE.DOLLY;
        }

        function handleMouseDownPan(event) {
            panStart.set(event.clientX, event.clientY);
            state = STATE.PAN;
        }

        function handleMouseMoveRotate(event) {
            rotateEnd.set(event.clientX, event.clientY);
            rotateDelta.subVectors(rotateEnd, rotateStart).multiplyScalar(0.005);
            sphericalDelta.theta -= rotateDelta.x;
            sphericalDelta.phi -= rotateDelta.y;
            rotateStart.copy(rotateEnd);
        }

        function handleMouseMoveDolly(event) {
            dollyEnd.set(event.clientX, event.clientY);
            dollyDelta.subVectors(dollyEnd, dollyStart);
            if (dollyDelta.y > 0) {
                scale *= 1.1;
            } else if (dollyDelta.y < 0) {
                scale /= 1.1;
            }
            dollyStart.copy(dollyEnd);
        }

        function handleMouseMovePan(event) {
            panEnd.set(event.clientX, event.clientY);
            panDelta.subVectors(panEnd, panStart);
            pan(panDelta.x, panDelta.y);
            panStart.copy(panEnd);
        }

        function handleMouseUp() {
            state = STATE.NONE;
        }

        function handleMouseWheel(event) {
            if (!scope.enableZoom) return;
            if (event.deltaY < 0) {
                scale /= 1.1;
            } else if (event.deltaY > 0) {
                scale *= 1.1;
            }
        }

        function handleTouchStart(event) {
            if (event.touches.length === 1) {
                rotateStart.set(event.touches[0].pageX, event.touches[0].pageY);
                state = STATE.ROTATE;
            } else if (event.touches.length === 2) {
                const dx = event.touches[0].pageX - event.touches[1].pageX;
                const dy = event.touches[0].pageY - event.touches[1].pageY;
                const distance = Math.sqrt(dx * dx + dy * dy);
                dollyStart.set(0, distance);
                state = STATE.DOLLY;
            }
        }

        function handleTouchMove(event) {
            if (state === STATE.ROTATE && event.touches.length === 1) {
                rotateEnd.set(event.touches[0].pageX, event.touches[0].pageY);
                rotateDelta.subVectors(rotateEnd, rotateStart).multiplyScalar(0.005);
                sphericalDelta.theta -= rotateDelta.x;
                sphericalDelta.phi -= rotateDelta.y;
                rotateStart.copy(rotateEnd);
            } else if (state === STATE.DOLLY && event.touches.length === 2) {
                const dx = event.touches[0].pageX - event.touches[1].pageX;
                const dy = event.touches[0].pageY - event.touches[1].pageY;
                dollyEnd.set(0, Math.sqrt(dx * dx + dy * dy));
                dollyDelta.subVectors(dollyEnd, dollyStart);
                if (dollyDelta.y > 0) scale *= 1.1;
                else if (dollyDelta.y < 0) scale /= 1.1;
                dollyStart.copy(dollyEnd);
            }
        }

        function handleTouchEnd() {
            state = STATE.NONE;
        }

        function pan(deltaX, deltaY) {
            if (!scope.enablePan) return;
            const offset = new THREE.Vector3();
            const element = scope.domElement === document ? scope.domElement.body : scope.domElement;
            const position = scope.object.position;
            offset.copy(position).sub(target);
            const targetDistance = offset.length();
            targetDistance *= Math.tan((scope.object.fov / 2) * Math.PI / 180.0);
            panLeft(2 * deltaX * targetDistance / element.clientHeight, scope.object.matrix);
            panUp(2 * deltaY * targetDistance / element.clientHeight, scope.object.matrix);
        }

        function panLeft(distance, objectMatrix) {
            const v = new THREE.Vector3();
            v.setFromMatrixColumn(objectMatrix, 0);
            v.multiplyScalar(-distance);
            panOffset.add(v);
        }

        function panUp(distance, objectMatrix) {
            const v = new THREE.Vector3();
            v.setFromMatrixColumn(objectMatrix, 1);
            v.multiplyScalar(distance);
            panOffset.add(v);
        }

        // event bindings
        this.domElement.addEventListener('contextmenu', (e) => e.preventDefault());
        this.domElement.addEventListener('mousedown', function (event) {
            if (!scope.enabled) return;
            event.preventDefault();
            if (event.button === 0 && !event.shiftKey) {
                if (scope.enableRotate) handleMouseDownRotate(event);
            } else if (event.button === 1) {
                if (scope.enableZoom) handleMouseDownDolly(event);
            } else if (event.button === 2 || event.shiftKey) {
                if (scope.enablePan) handleMouseDownPan(event);
            }
        });

        this.domElement.addEventListener('mousemove', function (event) {
            if (!scope.enabled) return;
            event.preventDefault();
            if (state === STATE.ROTATE) handleMouseMoveRotate(event);
            else if (state === STATE.DOLLY) handleMouseMoveDolly(event);
            else if (state === STATE.PAN) handleMouseMovePan(event);
        });

        this.domElement.addEventListener('mouseup', function () {
            if (!scope.enabled) return;
            handleMouseUp();
        });

        this.domElement.addEventListener('wheel', function (event) {
            if (!scope.enabled) return;
            event.preventDefault();
            handleMouseWheel(event);
        }, { passive: false });

        this.domElement.addEventListener('touchstart', function (event) {
            if (!scope.enabled) return;
            handleTouchStart(event);
        }, { passive: false });
        this.domElement.addEventListener('touchmove', function (event) {
            if (!scope.enabled) return;
            handleTouchMove(event);
        }, { passive: false });
        this.domElement.addEventListener('touchend', function () {
            if (!scope.enabled) return;
            handleTouchEnd();
        });
    };
})();
